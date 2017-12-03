import requests
import traceback
from bs4 import BeautifulSoup
import time
import datetime
import sqlite3
import re

sherdog_url = 'http://www.sherdog.com/'
max_tries = 3
sleep_time = 1

def clean_data(input_data):
    if isinstance(input_data, str):
        input_data =  re.sub(r'[/n]','', input_data)
        return ' '.join(input_data.split())
    else:
        return input_data

def build_db():
    with sqlite3.connect('mma_final.db') as conn:
        conn.execute('create table if not exists matches (fighter_id TEXT, opponent_id TEXT, event_date TEXT, result TEXT, result_method TEXT, result_round int, fight_type TEXT, UNIQUE (fighter_id, opponent_id, event_date))')
        conn.execute('create table if not exists fighter (fighter_id TEXT UNIQUE, fighter_name, dob TEXT)')
        conn.commit()

def read_sherdog_fighter_page_to_db(soup, fighter_id):
    opponent_id_list = []

    with sqlite3.connect('mma_final.db') as conn:
        try:
            name = soup.find_all('h1', {'itemprop':'name'})[0].text
            dob = soup.find_all('span', {'itemprop':'birthDate'})[0].text
        except:
            return []
        try:
            conn.execute('insert into fighter values (?, ?, ?)', (fighter_id, name, dob))
        except sqlite3.IntegrityError:
            pass
        print(fighter_id, name, dob)

        fight_history_tables = soup.find_all('div', {'class':'module fight_history'})
        for section in fight_history_tables:
            table = section.find('table')
            header = section.find('div', {'class':'module_header'})
            if table is None or header is None or '-' not in header.text:
                continue
            fight_type = clean_data(header.text.split('-')[1])

            rows = table.find_all('tr')
            for i in rows[1:]:
                try:
                    columns = i.find_all('td')
                    if len(columns) == 6:
                        result = columns[0].text
                        opponent_id = columns[1].find('a')['href']
                        event_date = columns[2].find('span', {'class':'sub_line'}).text
                        formatted_date = datetime.datetime.strptime(event_date, '%b / %d / %Y')
                        result_method = columns[3].text
                        round = columns[4].text
                        if '/fighter/' in opponent_id:
                            opponent_id_list.append(opponent_id)
                        try:
                            conn.execute('insert into matches values (?, ?, ?, ?, ?, ?, ?)', (fighter_id, opponent_id, str(formatted_date), result, result_method, round, fight_type))
                        except sqlite3.IntegrityError:
                            pass
                except:
                    traceback.print_exc()

        conn.commit()
    opponent_id_list = list(set(opponent_id_list))
    return opponent_id_list

def make_request(url, type_of_request = 'get', payload = None):
    s = requests.Session()
    s.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'}
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
    subpath = 'fighter/'
    soup = make_request(sherdog_url + subpath + f_id)
    return read_sherdog_fighter_page_to_db(soup, f_id)

def print_db_stats():
    with sqlite3.connect('mma_final.db') as conn:
        fighters = conn.execute('select distinct fighter_id from fighter').fetchall()
        fights = conn.execute('select * from matches').fetchall()

        print('num of fighters: {0}, num of fights: {1}'.format(len(fighters), len(fights)))

#bfs starting with current ufc champions
def get_fighter_list():
    initial_search_list = {'Stipe-Miocic-39537':False,
                           'Daniel-Cormier-52311':False,
                           'Georges-St-Pierre-3500':False,
                           'Tyron-Woodley-42605':False,
                           'Conor-McGregor-29688':False,
                           'Max-Holloway-38671':False,
                           'TJ-Dillashaw-62507':False,
                           'Demetrious-Johnson-45452':False,
                           'Cristiane-Justino-14477':False,
                           'Amanda-Nunes-31496':False,
                           'Rose-Namajunas-69083':False}


    while len([initial_search_list[i] for i in initial_search_list.keys() if not initial_search_list[i]])>0:
        print('len of fighter list:', len(initial_search_list))
        relevant_search_list = {i:initial_search_list[i] for i in initial_search_list.keys() if not initial_search_list[i]}
        resulting_list = {}
        for i, j in relevant_search_list.items():
            for k in search_fighter(i):
                if k not in initial_search_list.keys():
                    initial_search_list[k] = False
            initial_search_list[i] = True
            print_db_stats()

def main():
    build_db()
    get_fighter_list()

if __name__ == '__main__':
    main()