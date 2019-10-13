import requests
from bs4 import BeautifulSoup
import datetime
import random
import copy
from urllib.parse import urljoin
import pandas as pd
from common import sleep_random_amount, clean_text, pad_num, boxing_data_location
from private import box_rec_user_name, box_rec_password
import pickle
import traceback
import functools
import operator
import tqdm


base_url = 'http://boxrec.com'
fight_columns = ['division', 'boxer', 'lbs', 'w-l-d', 'last 6', 'rounds', 'opponent', 'lbs', 'w-l-d', 'last 6']


class Scraper():
    def __init__(self, min_sleep_time = 5.0, max_sleep_time = 10.0, use_cache = True):
        self.min_sleep_time = min_sleep_time
        self.max_sleep_time = max_sleep_time

        self.scrape_date = datetime.datetime.today()
        self.login()

        self.fighter_data = []
        self.fight_data = []

        if use_cache:
            try:
                with open('{}/fighter_cache.pkl'.format(boxing_data_location), 'rb') as f:
                    self.fighter_url_cache = pickle.load(f)
            except:
                try:
                    with open('{}/fighter_cache_backup.pkl'.format(boxing_data_location), 'rb') as f:
                        self.fighter_url_cache = pickle.load(f)
                except:
                    self.fighter_url_cache = dict()

            try:
                with open('{}/fight_cache.pkl'.format(boxing_data_location), 'rb') as f:
                    self.fight_url_cache = pickle.load(f)
            except:
                try:
                    with open('{}/fight_cache_backup.pkl'.format(boxing_data_location), 'rb') as f:
                        self.fight_url_cache = pickle.load(f)
                except:
                    self.fight_url_cache = dict()

            try:
                with open('{}/fighter_url_set.pkl'.format(boxing_data_location), 'rb') as f:
                    self.fighter_url_set = pickle.load(f)
            except:
                try:
                    with open('{}/fighter_url_set.pkl'.format(boxing_data_location), 'rb') as f:
                        self.fighter_url_set = pickle.load(f)
                except:
                    self.fighter_url_set = set()
        else:
            self.fighter_url_cache = dict()
            self.fight_url_cache = dict()
            self.fighter_url_set = set()


    def login(self):
        self.s = requests.Session()
        self.s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        }
        self.s.get('http://boxrec.com/en/login')
        sleep_random_amount(min_time=self.min_sleep_time, max_time=self.max_sleep_time)
        data = {'_target_path': '',
                '_username': box_rec_user_name,
                '_password': box_rec_password,
                'login[go]': ''}
        self.s.post('http://boxrec.com/en/login', data=data)
        sleep_random_amount(min_time=self.min_sleep_time, max_time=self.max_sleep_time)
        self.s.get(base_url)
        sleep_random_amount(min_time=self.min_sleep_time, max_time=self.max_sleep_time)

    def save_cache(self):
        with open('{}/fighter_cache.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fighter_url_cache, f)
        with open('{}/fighter_cache_backup.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fighter_url_cache, f)

        with open('{}/fight_cache.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fight_url_cache, f)
        with open('{}/fight_cache_backup.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fight_url_cache, f)

        with open('{}/fighter_url_set.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fighter_url_set, f)
        with open('{}/fighter_url_set_backup.pkl'.format(boxing_data_location), 'wb') as f:
            pickle.dump(self.fighter_url_set, f)

    def scrape_fights_at_url(self, url,  year, month, day):

        self.fight_url_cache.setdefault(url, [])

        r = self.s.get(url)
        soup = BeautifulSoup(r.text)
        table = soup.find('table', {'id':'calendarDate'})
        if table:
            records = table.find_all(['thead', 'tbody'], recusive = False)
        else:
            return []
        if not records:
            return []

        columns_mapping = dict()
        fighter_set = set()
        fight_counter = 0
        event_key = None
        for i in records:

            if i.name == 'thead' and i.find('a'):
                event_key = i.get_text()
            elif i.name == 'thead' and ('last 6' in i.get_text() or 'division' in i.get_text()):
                for c, t in enumerate(i.find_all('th')):
                    if 'division' in t.get_text():
                        columns_mapping['division'] = [c]
                    if 'boxer' in t.get_text():
                        columns_mapping['fighter1'] = [c]
                    if 'lbs' in t.get_text():
                        columns_mapping.setdefault('lbs', [])
                        columns_mapping['lbs'].append(c)
                    if 'w-l-d' in t.get_text():
                        columns_mapping.setdefault('w-l-d', [])
                        columns_mapping['w-l-d'].append(c)
                    if 'last 6' in t.get_text():
                        columns_mapping.setdefault('last 6', [])
                        columns_mapping['last 6'].append(c)
                    if 'opponent' in t.get_text():
                        columns_mapping.setdefault('fighter2', [])
                        columns_mapping['fighter2'].append(c)

            elif i.name == 'tbody' and columns_mapping:
                tr_tags = i.find_all('tr', recursive=False)
                notes = ''
                title_link = ''

                for j in reversed(tr_tags):

                    if j.find('a', {'class': 'titleLink'}):
                        notes = j.get_text()
                        title_link = j.find('a', {'class': 'titleLink'})['href']
                    elif not j.find('a'):
                        notes = j.get_text()
                    else:
                        notes_copy = copy.deepcopy(notes)
                        title_link_copy = copy.deepcopy(title_link)
                        notes = ''
                        title_link = ''
                        fight_id = j['id']
                        division = j.find_all('td')[columns_mapping['division'][0]].get_text().strip()
                        if not j.find_all('td')[columns_mapping['fighter1'][0]].find('a'):
                            continue
                        fighter1_name = j.find_all('td')[columns_mapping['fighter1'][0]].find('a').get_text().strip()
                        fighter1_url = j.find_all('td')[columns_mapping['fighter1'][0]].find('a')['href'].strip()
                        fighter1_record = j.find_all('td')[columns_mapping['w-l-d'][0]].get_text().strip()
                        last_6_1 = j.find_all('td')[columns_mapping['last 6'][0]].get_text().strip()
                        lbs1 = j.find_all('td')[columns_mapping['lbs'][0]].get_text().strip()

                        result = j.find_all('td')[columns_mapping['last 6'][0] + 1].get_text().strip()
                        result_method = j.find_all('td')[columns_mapping['last 6'][0] + 2].get_text().strip()

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

                        if not j.find_all('td')[columns_mapping['fighter2'][0]].find('a'):
                            continue
                        fighter2_name = j.find_all('td')[columns_mapping['fighter2'][0]].find('a').get_text().strip()
                        fighter2_url = j.find_all('td')[columns_mapping['fighter2'][0]].find('a')['href'].strip()
                        fighter2_record = j.find_all('td')[columns_mapping['w-l-d'][1]].get_text().strip()
                        last_6_2 = j.find_all('td')[columns_mapping['last 6'][1]].get_text().strip()
                        lbs2 = j.find_all('td')[columns_mapping['lbs'][1]].get_text().strip()

                        try:
                            bout_link = j.find_all('a')[-2]['href']
                            wiki_link = j.find_all('a')[-1]['href']
                        except:
                            traceback.print_exc()

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
                                    'day':day,
                                    'last_6':last_6_1,
                                    'lbs':lbs1,
                                    'result_method':result_method}

                        new_rec2 = {'event': event_key,
                                    'notes': notes_copy,
                                    'title_link': title_link_copy,
                                    'id': fight_id,
                                    'division': division,
                                    'fighter_name': fighter2_name,
                                    'fighter_url': fighter2_url,
                                    'fighter_record': fighter2_record,
                                    'result': result2,
                                    'opponent_name': fighter1_name,
                                    'opponent_url': fighter1_url,
                                    'opponent_record': fighter1_record,
                                    'bout_link': bout_link,
                                    'wiki_link': wiki_link,
                                    'year':year,
                                    'month':month,
                                    'day':day,
                                    'last_6':last_6_2,
                                    'lbs':lbs2,
                                    'result_method':result_method}

                        fight_counter += 1

                        self.fighter_url_set.add(fighter1_url)
                        self.fighter_url_set.add(fighter2_url)
                        fighter_set.add(fighter2_url)
                        fighter_set.add(fighter1_url)
                        self.fight_url_cache[url].append(new_rec1)
                        self.fight_url_cache[url].append(new_rec2)

                        # self.fight_data.append(new_rec1)
                        # self.fight_data.append(new_rec2)
        print('Found {0} fights and {1} fighters at {2}'.format(fight_counter, len(fighter_set), url))
        self.scrape_fighters()
        print('Current data count', len(self.fight_data), len(self.fighter_data))
        print()


    def scrape_fighters(self):
        self.fighter_data = []

        # print(' num of fighters: {0}'.format(len(self.fighter_url_set)))
        num_of_website_hits = 0
        for i in tqdm.tqdm(self.fighter_url_set):
            # if max_website_hits > 0 and num_of_website_hits >= max_website_hits:
            #     break
            try:
                if i not in self.fighter_url_cache:
                    # print('scraping fighter: {0}'.format(i), num_of_website_hits,
                    #       len(self.fighter_url_set), len(self.fight_data),
                    #       len(self.fighter_url_set) / len(self.fight_data),
                    #       len(self.fighter_url_cache))
                    # print('scraping: {0}, {1}, {2}'.format(i, c, num_of_website_hits))

                    num_of_website_hits += 1
                    new_output = {'fighter_url': i}
                    r = self.s.get(i)
                    sleep_random_amount(min_time=self.min_sleep_time, max_time=self.max_sleep_time)
                    soup = BeautifulSoup(r.text)
                    table = soup.find('table', {'class':'profileTable'})
                    table = table.find_all('td', {'class':'profileTable'})[1]
                    tables = table.find_all('table', {'class':'rowTable'})

                    for t in tables:
                        for r in t.find_all('tr'):
                            td_tags = r.find_all('td')
                            if len(td_tags) == 2:
                                new_output[td_tags[0].get_text().strip()] = td_tags[1].get_text().strip()
                    self.fighter_url_cache[i] = new_output
            except:
                traceback.print_exc()
                sleep_random_amount(min_time=60, max_time=300)

        for i in self.fighter_url_cache:
            if i not in self.fighter_data:
                self.fighter_data.append(self.fighter_url_cache[i])

        self.fighter_data = [j for _, j in self.fighter_url_cache.items()]
        self.fight_data = functools.reduce(operator.concat, [i for i in self.fight_url_cache.values()])
        self.save_cache()

    def save_data(self):
        self.save_cache()

        df1 = pd.DataFrame.from_dict(self.fight_data)
        df2 = pd.DataFrame.from_dict(self.fighter_data)

        df1 = df1.applymap(clean_text)
        df2 = df2.applymap(clean_text)

        df1 = df1.drop_duplicates()
        df2 = df2.drop_duplicates()

        df1.to_csv('{0}/fights_{1}_{2}_{3}.csv'.format(boxing_data_location, pad_num(self.scrape_date.year, 4),
                                                       pad_num(self.scrape_date.month, 2),
                                                       pad_num(self.scrape_date.day, 2)), sep='|', index=False)
        df1.to_csv('{0}/fights_{1}_{2}_{3}_backup.csv'.format(boxing_data_location, pad_num(self.scrape_date.year, 4),
                                                       pad_num(self.scrape_date.month, 2),
                                                       pad_num(self.scrape_date.day, 2)), sep='|', index=False)
        df2.to_csv('{0}/fighters_{1}_{2}_{3}.csv'.format(boxing_data_location, pad_num(self.scrape_date.year, 4),
                                                         pad_num(self.scrape_date.month, 2),
                                                         pad_num(self.scrape_date.day, 2)), sep='|', index=False)
        df2.to_csv('{0}/fighters_{1}_{2}_{3}_backup.csv'.format(boxing_data_location, pad_num(self.scrape_date.year, 4),
                                                         pad_num(self.scrape_date.month, 2),
                                                         pad_num(self.scrape_date.day, 2)), sep='|', index=False)
        print(df1.shape, df2.shape)
        self.login()


    def scrape_website(self, min_year, max_year, date_chunk_size = 100, randomize = True):
        date_list = []
        start_date = datetime.date(min_year, 1, 1)
        end_date = datetime.date(max_year, 12, 31)
        new_date = end_date

        while new_date >= start_date:
            if new_date < datetime.date.today():
                date_list.append(new_date)
            new_date = new_date - datetime.timedelta(days=1)

        if randomize:
            random.shuffle(date_list)

        for c, i in enumerate(date_list):
            try:
                year_str = pad_num(i.year, 4)
                month_str = pad_num(i.month, 2)
                day_str = pad_num(i.day, 2)
                # print(c, 'http://boxrec.com/en/date?date={0}-{1}-{2}'.format(year_str, month_str, day_str),
                #       len(self.fight_data))
                url = 'http://boxrec.com/en/date?date={0}-{1}-{2}'.format(year_str, month_str, day_str)

                if url not in self.fight_url_cache:
                    self.scrape_fights_at_url(url, year_str, month_str, day_str)
                    sleep_random_amount(min_time=self.min_sleep_time, max_time=self.max_sleep_time)
            except:
                traceback.print_exc()
                sleep_random_amount(min_time=60, max_time=300)

            if c % date_chunk_size == 0:
                self.save_data()


if __name__ == '__main__':
    use_cache = True
    s = Scraper(min_sleep_time = .1, max_sleep_time = .5, use_cache = use_cache)
    s.scrape_website(2000, 2019, date_chunk_size = 1)


