import requests
import traceback
from bs4 import BeautifulSoup
import time
import datetime
import sqlite3
import re
import random
import configparser
import configparser


tapology_url = 'http://www.tapology.com/'
max_tries = 3
sleep_time = .1

config = configparser.ConfigParser()
config.read('properties.ini')
db_location = config.get('mma', 'db_location')

def clean_data(input_data):
    if isinstance(input_data, str):
        input_data =  re.sub(r'[\n]','', input_data)
        return ' '.join(input_data.split())
    else:
        return input_data

def build_db():
    with sqlite3.connect(db_location) as conn:
        conn.execute(r'create table if not exists matches (fighter_id TEXT, name TEXT, dob TEXT, birth_loc TEXT, '
                     r'height TEXT, reach TEXT, result TEXT, method TEXT, details TEXT, opponent_name TEXT, '
                     r'opponent_id TEXT, event_date TEXT, f_record TEXT, o_record TEXT, fight_id TEXT,fight_division TEXT, fight_weight TEXT, '
                     r'UNIQUE (fighter_id, opponent_id, event_date))')

        # conn.execute(r'create table if not exists fighter (fighter_id TEXT UNIQUE, fighter_name, dob TEXT)')
        conn.commit()

def read_tapology_fighter_page_to_db(soup, fighter_id):
    opponent_id_list = []
    with sqlite3.connect(db_location) as conn:
        print(fighter_id)
        try:
            stats = soup.find('div', {'id':'stats'})
            li_tags = stats.find_all('li')
            name = clean_data(li_tags[0].find('span').text)
        except:
            return []
        try:
            dob = clean_data(li_tags[4].find_all('span')[-1].text)
            birth_loc = clean_data(li_tags[10].find_all('span')[-1].text)
        except:
            dob = ''
            birth_loc = ''

        try:
            height = clean_data(li_tags[8].find_all('span')[0].text.split('(')[1].split('cm')[0])
            reach = clean_data(li_tags[8].find_all('span')[1].text.split('(')[1].split('cm')[0])
        except:
            height = ''
            reach = ''

        results_by_type = soup.findAll('section', {'class', 'fighterFightResults'})

        for result_type in results_by_type:
            for fight_soup in result_type.find('ul').findAll('li'):
                try:

                    f_res = fight_soup.find('div', {'class':'result'})

                    opponent_id = f_res.find('div', {'class':'name'}).find('a')

                    try:
                        fight_id = fight_soup.find('a', {'title':'Bout Page'})['href']
                    except:
                        fight_id = None

                    if opponent_id:
                        opponent_name = opponent_id.text
                        opponent_id = opponent_id['href']
                        opponent_id_list.append(opponent_id)
                    else:
                        opponent_name = ''
                        opponent_id = ''

                    if f_res.find('span', {'title':'Fighter Record Before Fight'}):
                        f_record = f_res.find('span', {'title':'Fighter Record Before Fight'}).text
                    else:
                        f_record = ''
                    if f_res.find('span', {'title': 'Opponent Record Before Fight'}):
                        o_record = f_res.find('span', {'title':'Opponent Record Before Fight'}).text
                    else:
                        o_record = ''

                    f_date_str = f_res.find('div', {'class':'date'}).text
                    f_date = datetime.datetime.strptime(f_date_str, '%Y.%m.%d')

                    if fight_soup.find('div', {'class':'lead'}).find('a'):
                        f_result = fight_soup.find('div', {'class':'lead'}).find('a').text
                    else:
                        f_result = ''
                    if len(f_result.split('·')) == 1:
                        continue
                    win_loss_draw = None
                    if 'Win' in f_result:
                        win_loss_draw = 'Win'
                    elif 'Loss' in  f_result:
                        win_loss_draw = 'Loss'
                    elif 'Draw' in f_result:
                        win_loss_draw = 'Draw'
                    else:
                        win_loss_draw = f_result.split('·')[0]

                    try:
                        fight_weight = fight_soup.find('span', {'class': 'weigh-in'}).text
                    except:
                        fight_weight = None

                    try:
                        fight_division = fight_soup.find('div', {'class':'weight'}).text
                        fight_division = fight_division.replace(fight_weight, '')
                        fight_division = fight_division.split(' ')[-2]
                        # fight_division = ''.join(re.findall('\d+', fight_division))

                        fight_weight = float(fight_weight.split('(')[1].split('kgs')[0])
                    except:
                        fight_division = None


                    if not win_loss_draw:
                        continue

                    method = f_result.split('·')[1]

                    print(fighter_id, name, dob, birth_loc, height, reach, win_loss_draw, method,
                                      f_result, opponent_name, opponent_id, f_date, f_record, o_record, fight_id,
                                      fight_division, fight_weight)
                    try:
                        conn.execute('insert into matches values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                     (fighter_id, name, dob, birth_loc, height, reach, win_loss_draw, method,
                                      f_result, opponent_name, opponent_id, f_date, f_record, o_record, fight_id,
                                      fight_division, fight_weight))
                    except sqlite3.IntegrityError:
                        # traceback.print_exc()
                        pass
                    except:
                        traceback.print_exc()
                except:
                    traceback.print_exc()

            conn.commit()

    opponent_id_list = list(set(opponent_id_list))
    print(opponent_id_list)
    return opponent_id_list

def make_request(url, type_of_request = 'get', payload = None):
    s = requests.Session()
    s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
    for i in range(max_tries):
        time.sleep(sleep_time)
        try:
            r = s.get(url)
            return BeautifulSoup(r.text, 'html.parser')
        except:
            traceback.print_exc()
            time.sleep(sleep_time)
    return BeautifulSoup('<body/>')

def search_fighter(f_id):
    soup = make_request(tapology_url +  f_id)
    return read_tapology_fighter_page_to_db(soup, f_id)

def print_db_stats():
    with sqlite3.connect(db_location) as conn:
        fights = conn.execute('select * from matches').fetchall()

        print(' num of fights: {0}'.format( len(fights)))


def get_initial_fighters():
    men_url = 'https://www.tapology.com/rankings/top-ten-fan-favorite-mma-and-ufc-fighters'
    #female_url = 'https://www.tapology.com/rankings/33-current-best-pound-for-pound-female-mma-fighters'
    s = requests.Session()

    s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'

    r = s.get(men_url)
    soup = BeautifulSoup(r.text)
    header = soup.find_all('div', {'class':'rankingItemsItemRow name'})
    links1 = [i.find('a')['href'] for i in header if i.find('a')]

    return {i:False for i in  links1}

def get_found_fighters():
    with sqlite3.connect(db_location) as conn:
        opponent_ids = conn.execute('Select opponent_id from matches')

    f_dict = dict()
    for i in opponent_ids:
        f_dict[i[0]] = False
    return f_dict

#dfs starting with current ufc champions
def get_fighter_list():
    initial_search_list = get_initial_fighters()
    # found_fighter_list = get_found_fighters()
    # initial_search_list.update(found_fighter_list)

    while len([initial_search_list[i] for i in initial_search_list.keys() if not initial_search_list[i]])>0:
        relevant_search_list = [i for i in initial_search_list.keys() if not initial_search_list[i]]
        print('fighters to search through:', len(relevant_search_list))
        random.shuffle(relevant_search_list)

        #bfs
        new_urls = []
        for count, i in enumerate(relevant_search_list):
            new_urls.extend(search_fighter(i))
            initial_search_list[i] = True
            print_db_stats()
            new_urls = list(set(new_urls))
            print(count, len(relevant_search_list))
            print('new fighters to search through:', len(new_urls))
        for i in new_urls:
            if i not in initial_search_list.keys():
                initial_search_list[i] = False
        # #dfs
        # chosen_url = random.choice(relevant_search_list)
        # new_urls = search_fighter(chosen_url)
        #
        # for i in new_urls:
        #     if i not in initial_search_list.keys():
        #         initial_search_list[i] = False
        # initial_search_list[chosen_url] = True
        # print_db_stats()


def main():
    while True:
        try:
            build_db()
            get_fighter_list()
        except:
            traceback.print_exc()
            time.sleep(100)

if __name__ == '__main__':
    main()