import requests
import datetime
from common import pad_num, sleep_random_amount, get_session, get_soup
from bs4 import BeautifulSoup
import sqlite3
import pickle
import pandas as pd
import logging

base_url = 'https://www.basketball-reference.com/'
day_scores_base_url = 'https://www.basketball-reference.com/boxscores/?month={month}6&day={day}&year={year}'
data_path = r'C:\Users\trist\Documents\nba_data'
db_name = 'nba_db'
box_score_link_table_name = 'boxscore_links'
box_score_details_table_name = 'boxscore_details'

class Scraper:
    def __init__(self, start_date = None, end_date = None):

        self.start_date = start_date
        if not end_date:
            self.end_date = datetime.date.today()
            self.current_date = datetime.date.today()

        if not start_date:
            self.start_date = datetime.date(1980, 1, 1)

        self.session = get_session()
        self.box_office_links = pd.DataFrame()
        self.box_office_details = pd.DataFrame()
        self.create_dbs()
        self.load_data()

    def load_data(self):
        with sqlite3.connect('{data_path}/{db_name}.db'.format(data_path=data_path, db_name=db_name)) as conn:
            self.box_office_links = pd.read_sql('''Select * from {table_name}'''.format(table_name=box_score_link_table_name), conn)
        with sqlite3.connect('{data_path}/{db_name}.db'.format(data_path=data_path, db_name=db_name)) as conn:
            self.box_office_details = pd.read_sql('''Select * from {table_name}'''.format(table_name=box_score_details_table_name), conn)

    def create_dbs(self):
        with sqlite3.connect('{data_path}/{db_name}.db'.format(data_path=data_path, db_name=db_name)) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS {table_name} (url text PRIMARY KEY, year text, month text, day text);'.format(table_name=box_score_link_table_name))
            conn.execute('CREATE TABLE IF NOT EXISTS {table_name} (url text PRIMARY KEY, year text, month text, day text);'.format(table_name=box_score_details_table_name))

    def scrape_current_day_boxscore_links(self):
        padded_year = pad_num(self.current_date.year, 4)
        padded_month = pad_num(self.current_date.month, 2)
        padded_day = pad_num(self.current_date.day, 2)

        url = day_scores_base_url.format(month = padded_year,
                                   day = padded_month,
                                   year = padded_day)
        soup = get_soup(url, session = self.session)
        games = soup.find_all('p', {'class': 'links'})

        for i in games:
            links = i.find_all('a')
            days_games_box_score_links = [j['href'] for j in links if 'boxscores' in j['href']]
            self.game_links.extend(days_games_box_score_links)

        with sqlite3.connect('{data_path}/{db_name}.db'.format(data_path=data_path, db_name=db_name)) as conn:
            try:
                conn.execute(
                    'insert into {table_name} ({url}, {year}, {month}, {day});'.format(table_name = box_score_link_table_name,
                                                                                                   url = url,
                                                                                                   year = padded_year,
                                                                                                   month = padded_month,
                                                                                                   day = padded_day))
            except sqlite3.IntegrityError:
                logging.exception('integrity error')

    def scrape_date_range_boxscore_links(self):
        while self.current_date <= self.end_date and self.current_date >= self.start_date:
            self.current_date -= datetime.timedelta(days=1)
            self.scrape_current_day_boxscore_links()
        self.load_data()

    def scrape_box_office_details(self):
        pass

    def scrape_all_box_office_details(self):
        box_links = set(self.box_office_links['url'])
        for i in box_links:
            url = base_url + i
            soup = get_soup(url, session = self.session)
            





if __name__ == '__main__':
    scraper = Scraper()
    scraper.scrape_date_range_boxscore_links()
