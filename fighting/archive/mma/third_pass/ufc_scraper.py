import requests
from bs4 import BeautifulSoup
import os
import pickle
import time
import json
import random
import traceback
from collections import OrderedDict
import pandas as pd

db_loc = r"C:/Users/trist/Documents/db_loc/mma/db/ufc/"
fighter_set_loc = 'fighter_set.plk'
event_set_loc = 'event_set.plk'
fight_details_loc = 'ufc_fight_data.plk'

def read_event(event_url):
    data = []
    s = requests.Session()
    r = s.get('http://www.ufc.com' + event_url)
    soup = BeautifulSoup(r.text, "html5lib")
    event_id = r.text.split(r'http://liveapi.fightmetric.com/V1/')[1].split('/')[0]

    panels = soup.find_all('a', {'target':'_self'})
    keys = []
    for i in panels:
        if i and i.find('div', {'class':'fight'}):
            keys.append(i.find('div', {'class':'fight'})['data-fight-stat-id'])

    for k in keys:
        r = s.get('http://liveapi.fightmetric.com/V2/{0}/{1}/Stats.json'.format(event_id, k))
        j = json.loads(r.text)
        print(j)
        for fighter_key in j['FMLiveFeed']['FightStats'].keys():
            fighter_id = j['FMLiveFeed']['FightStats'][fighter_key]['FighterID']

            f_dict = OrderedDict()
            f_dict['Name'] = None
            for fighter_side_key, fighter_side_value in j['FMLiveFeed']['Fighters'].items():
                if fighter_side_value['FighterID'] == fighter_id:
                    f_dict['Name'] = fighter_side_value['Name'].lower()

            f_dict['EventID'] = j['FMLiveFeed']['EventID']
            f_dict['FightID'] = j['FMLiveFeed']['FightID']
            f_dict['WeightClass'] = j['FMLiveFeed']['WeightClass']
            f_dict['Timestamp'] = j['Timestamp']

            for strike_stat_key, strike_stat_value in j['FMLiveFeed']['FightStats'][fighter_key]['Strikes'].items():
                for strike_state_type_key, strike_state_type_value in strike_stat_value.items():
                    key_name = strike_stat_key + ' ' + strike_state_type_key
                    key_name = key_name.replace(' ', '_')
                    f_dict[key_name] = strike_state_type_value
            for strike_stat_key, strike_stat_value in j['FMLiveFeed']['FightStats'][fighter_key]['Grappling'].items():
                for strike_state_type_key, strike_state_type_value in strike_stat_value.items():
                    key_name = strike_stat_key + ' ' + strike_state_type_key
                    key_name = key_name.replace(' ', '_')
                    f_dict[key_name] = strike_state_type_value
            data.append(f_dict)
    return data


def get_champions():
    url = 'http://www.ufc.com/fighters'
    s = requests.Session()
    r = s.get(url)
    soup = BeautifulSoup(r.text, "html5lib")
    fighter_links = [i.find('a')['href'].lower() for i in soup.find_all('div', {'class':'wc-column title-column'})]
    return fighter_links


def get_event_results(soup, f_url):
    fight_soups = soup.find_all('tr', {'class':'fight'}) + soup.find_all('tr', {'class':'fight moreFights'})
    opponent_urls = []
    event_urls = []
    fighter_result_data = []

    name = soup.find('meta', {'property':'og:title'})['content']
    for i in fight_soups:
        if i.find('td', {'class':'fighter'}) and i.find('td', {'class':'result'}):
            event_url, opponent_url = None, None
            if i.find('td', {'class':'fighter'}).find('a'):
                opponent_urls.append(i.find('td', {'class':'fighter'}).find('a')['href'].lower())
                opponent_url = i.find('td', {'class':'fighter'}).find('a')['href'].lower()
            else:
                opponent_url = None


            if i.find('td', {'class':'event'}).find('a'):
                event_urls.append(i.find('td', {'class':'event'}).find('a')['href'].lower())
                event_url = i.find('td', {'class':'event'}).find('a')['href'].lower()
            else:
                event_url = None
            result = None
            try:
                result = i.find('td', {'class':'result'}).find('td').text
            except:
                if 'win' in str(i).lower():
                    result = 'win'
                elif 'draw' in str(i).lower():
                    result = 'draw'
                elif ('no' in str(i).lower() and 'contest' in str(i).lower()) or 'nc' in str(i).lower():
                    result = 'nc'
            method_text = ' '.join(i.find('td', {'class':'method'}).text.split())
            fighter_result_data.append({'f_url':f_url,
                                        'e_url':event_url,
                                        'result':result,
                                        'method_text':method_text,
                                        'o_url':opponent_url,
                                        'name':name.lower()})
    return opponent_urls, event_urls, fighter_result_data


def get_opponent_urls(fighter_url):
    s = requests.Session()
    s.get('http://www.ufc.com')
    time.sleep(2)
    r = s.get('http://www.ufc.com' + fighter_url, timeout = 30)
    soup = BeautifulSoup(r.text, "html5lib")
    opponent_urls, event_urls, fighter_result_data = get_event_results(soup, fighter_url)
    return event_urls, opponent_urls, fighter_result_data


def get_fighters_and_events():
    try:
        event_urls_total, fighter_urls = None, None
        df = pd.read_pickle(os.path.join(db_loc, fighter_set_loc))
    except:
        traceback.print_exc()
        fighter_urls = get_champions()
        event_urls_total = set()
        fighter_result_dict = []

        searched_urls = set()
        while len(fighter_urls) > 0:
            try:
                next_url = fighter_urls.pop()
                searched_urls.add(next_url)
                event_urls, opponent_urls, fighter_result_data = get_opponent_urls(next_url)
                fighter_result_dict.extend(fighter_result_data)
                fighter_urls.extend([i for i in opponent_urls if i not in searched_urls])
                fighter_urls = list(set(fighter_urls))
                event_urls_total.update(set(event_urls))
                print(next_url, len(fighter_urls), len(event_urls_total))
            except:
                traceback.print_exc()
                time.sleep(2)
        df = pd.DataFrame.from_dict(fighter_result_dict)
        df.to_pickle(os.path.join(db_loc, fighter_set_loc))
    return df


def scrape():
    fighter_df = get_fighters_and_events()
    event_urls = set(fighter_df['e_url'].tolist())
    max_tries = 5
    data = []
    for i in event_urls:
        for _ in range(max_tries):
            try:
                data.extend(read_event(i))
                break
            except:
                time.sleep(10)
    fight_details_df = pd.DataFrame.from_dict(data)
    fight_details_df.to_pickle(os.path.join(db_loc, fight_details_loc))


def get_data():
    df = pd.read_pickle(os.path.join(db_loc, fighter_set_loc))
    fight_details_df = pd.read_pickle(os.path.join(db_loc, fight_details_loc))
    print()


if __name__ == '__main__':
    scrape()