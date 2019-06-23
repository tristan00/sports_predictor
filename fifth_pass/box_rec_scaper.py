import requests
from bs4 import BeautifulSoup
import datetime
import random
import copy
from urllib.parse import urljoin
from scipy import stats
import time
import pandas as pd
from common import sleep_random_amount, clean_text, pad_num, boxing_data_location, box_rec_user_name, box_rec_password
from private import box_rec_user_name, box_rec_password

base_url = 'http://boxrec.com'

fighter_url_cache = dict()

class Scraper():
    def __init__(self):
        self.s = requests.Session()
        self.s.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    }
        self.s.get('http://boxrec.com/en/login')

        data = {'_target_path':'',
                '_username':box_rec_user_name,
                '_password':box_rec_password,
                'login[go]':''}
        self.s.post('http://boxrec.com/en/login')

    def scrape_website(self, min_year, max_year):
        self.fights_output = []
        self.fighters_output = []

        for i in range(max_year, min_year, -1):
            f1, f2 = scrape_date_range(datetime.date(i, 1, 1), datetime.date(i, 12, 31))
            self.fights_output.append(f1)
            self.fighters_output.append(f2)

            df1 = pd.DataFrame.from_dict(self.fights_output)
            df2 = pd.DataFrame.from_dict(self.fighters_output)

            df1 = df1.applymap(clean_text)
            df2 = df2.applymap(clean_text)

            df1 = df1.drop_duplicates()
            df2 = df2.drop_duplicates()

            df1.to_csv('{}/fights.csv'.format(boxing_data_location), sep='|', index=False)
            df2.to_csv('{}/fighters.csv'.format(boxing_data_location), sep='|', index=False)
            print(df1.shape, df2.shape)



def scrape_fights_at_date(year, month, day):
    year_str = pad_num(year, 4)
    month_str = pad_num(month, 2)
    day_str = pad_num(day, 2)
    print('http://boxrec.com/en/date?date={0}-{1}-{2}'.format(year_str, month_str, day_str))
    r = requests.get('http://boxrec.com/en/date?date={0}-{1}-{2}'.format(year_str, month_str, day_str))
    soup = BeautifulSoup(r.text)
    table = soup.find('table', {'id':'calendarDate'})
    if table:
        records = table.find_all(['thead', 'tbody'], recusive = False)
    else:
        return []
    if not records:
        return []

    output = []
    event_key = None
    for i in records:
        if i.name == 'thead' and i.find('a'):
            event_key = i.get_text()

        elif i.name == 'tbody':

            tr_tags = i.find_all('tr', recursive=False)
            notes = ''
            title_link = ''

            for j in reversed(tr_tags):

                if j.find('a', {'class': 'titleLink'}):
                    notes = j.get_text()
                    title_link = j.find('a', {'class': 'titleLink'})['href']
                else:
                    notes_copy = copy.deepcopy(notes)
                    title_link_copy = copy.deepcopy(title_link)
                    notes = ''
                    title_link = ''

                    fight_id = j['id']
                    division = j.find_all('td')[1].get_text().strip()
                    fighter1_name = j.find_all('td')[2].find('a').get_text().strip()
                    fighter1_url = j.find_all('td')[2].find('a')['href'].strip()
                    fighter1_record = j.find_all('td')[3].get_text().strip()
                    result = j.find_all('td')[5].get_text().strip()

                    if result.upper() == 'W':
                        result1 = 'W'
                        result2 = 'L'
                    elif result.upper() == 'L':
                        result1 = 'L'
                        result2 = 'W'
                    elif result.upper() == 'D':
                        result1 = 'D'
                        result2 = 'D'
                    elif result.upper() == 'N':
                        result1 = 'N'
                        result2 = 'N'
                    else:
                        continue

                    fighter2_name = j.find_all('td')[6].find('a').get_text().strip()
                    fighter2_url = j.find_all('td')[6].find('a')['href'].strip()
                    fighter2_record = j.find_all('td')[7].get_text().strip()

                    bout_link = j.find_all('td')[10].find_all('a')[0]['href']
                    wiki_link = j.find_all('td')[10].find_all('a')[1]['href']

                    if title_link_copy:
                        title_link_copy = urljoin(base_url, title_link_copy)
                    fighter1_url = urljoin(base_url, fighter1_url)
                    fighter2_url = urljoin(base_url, fighter2_url)
                    bout_link = urljoin(base_url, bout_link)
                    wiki_link = urljoin(base_url, wiki_link)

                    new_rec1 = {'event': event_key,
                                'notes': notes_copy,
                                'title_link': title_link_copy,
                                'id': fight_id,
                                'division':division,
                                'fighter_name':fighter1_name,
                                'fighter_url':fighter1_url,
                                'fighter_record':fighter1_record,
                                'result':result1,
                                'opponent_name':fighter2_name,
                                'opponent_url':fighter2_url,
                                'opponent_record':fighter2_record,
                                'bout_link':bout_link,
                                'wiki_link':wiki_link,
                                'year':year,
                                'month':month,
                                'day':day}

                    new_rec2 = {'event': event_key,
                                'notes': notes_copy,
                                'title_link': title_link_copy,
                                'id': fight_id,
                                'division': division,
                                'fighter_name': fighter2_name,
                                'fighter_url': fighter2_url,
                                'fighter_record': fighter2_record,
                                'result': result2,
                                'opponent_name': fighter1_url,
                                'opponent_url': fighter1_url,
                                'opponent_record': fighter1_record,
                                'bout_link': bout_link,
                                'wiki_link': wiki_link,
                                'year':year,
                                'month':month,
                                'day':day}
                    output.append(new_rec1)
                    output.append(new_rec2)
    return output


def scrape_fighters(url_list):
    global fighter_url_cache

    output = []

    for c, i in enumerate(url_list):
        print(c, len(url_list), len(output))
        if i not in fighter_url_cache:
            new_output = {'fighter_url': i}
            r = requests.get(i)
            sleep_random_amount()
            soup = BeautifulSoup(r.text)
            table = soup.find('table', {'class':'profileTable'})
            table = table.find_all('td', {'class':'profileTable'})[1]
            tables = table.find_all('table', {'class':'rowTable'})

            for t in tables:
                for r in t.find_all('tr'):
                    td_tags = r.find_all('td')
                    new_output[td_tags[0].get_text().strip()] = td_tags[1].get_text().strip()
            fighter_url_cache[i] = new_output
        output.append(fighter_url_cache[i])

    return output


def scrape_date_range(start_day, end_day, randomize = True):
    date_list = []

    new_date = end_day

    while new_date >= start_day:
        if new_date < datetime.date.today():
            date_list.append(new_date)
        new_date = new_date - datetime.timedelta(days=1)

    if randomize:
        random.shuffle(date_list)
    print('generated date list: {0}'.format(date_list))

    fights = []
    for c, i in enumerate(date_list):
        print('starting date: {0}'.format(i))
        fights.extend(scrape_fights_at_date(i.year, i.month, i.day))
        sleep_random_amount()
        print(i, c, len(date_list), len(date_list) - c, len(fights))

    fighter_urls = list(set([i['fighter_url'] for i in fights]))
    fighters = scrape_fighters(fighter_urls)
    return fights, fighters





if __name__ == '__main__':
    s = Scraper()
    s.scrape_website(1950, 2019)



