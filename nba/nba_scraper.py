import datetime
from nba.common import (sleep_on_error,
                    sleep_normal,
                    get_session,
                    base_url,
                    day_scores_base_url,
                    data_path,
                    box_score_link_table_name,
                    box_score_details_table_name,
                    player_detail_table_name,
                    date_record_pickle_file_name,
                    box_score_record_pickle_file_name,
                    max_tries,
                    file_lock)

from bs4 import BeautifulSoup
import pickle
import pandas as pd
import copy
import time
import traceback
import re

def get_soup(url, session = None, sleep = True):
    if sleep:
        sleep_normal()

    if not session:
        session = get_session()

    r = session.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    return soup


def process_stats_tables(t_basic, t_advanced):
    tbody_basic = t_basic.find('tbody')
    rows_basic = tbody_basic.find_all('tr')
    footer_basic = t_basic.find('tfoot')
    tbody_advanced = t_advanced.find('tbody')
    rows_advanced = tbody_advanced.find_all('tr')
    footer_advanced = t_advanced.find('tfoot')

    player_data = dict()
    team_data = dict()

    td_tags = footer_basic.find_all('td')
    for td_tag in td_tags:
        stat_name = td_tag['data-stat']
        stat_value = td_tag.get_text()
        team_data[stat_name] = stat_value
    td_tags = footer_advanced.find_all('td')
    for td_tag in td_tags:
        stat_name = td_tag['data-stat']
        stat_value = td_tag.get_text()
        team_data[stat_name] = stat_value

    for r in rows_basic:
        if len(r.find_all('th')) != 1:
            continue

        player_info = r.find('th')
        player_link = base_url + player_info.find('a')['href']
        player_data.setdefault(player_link, dict())
        player_data[player_link]['player_link'] = player_link
        player_data[player_link]['player_name'] = player_info.find('a').get_text()

        td_tags = r.find_all('td')
        for td_tag in td_tags:
            stat_name = td_tag['data-stat']
            stat_value = td_tag.get_text()
            player_data[player_link][stat_name] = stat_value

    for r in rows_advanced:
        if len(r.find_all('th')) != 1:
            continue

        player_info = r.find('th')
        player_link = base_url + player_info.find('a')['href']
        player_data.setdefault(player_link, dict())

        td_tags = r.find_all('td')
        for td_tag in td_tags:
            stat_name = td_tag['data-stat']
            stat_value = td_tag.get_text()
            player_data[player_link][stat_name] = stat_value
    return {'player_data': player_data, 'team_data': team_data}


def get_score_table(soup, tag, simplicity):
    pattern = 'box[-_]+{tag}.+{simplicity}'.format(tag = tag.upper(), simplicity = simplicity)
    table = soup.find('table', {'id':re.compile(pattern)})
    return table


class Scraper:
    def __init__(self, start_date = None, end_date = None, clear_data = False):
        self.end_date = end_date
        self.current_date = end_date
        self.start_date = start_date
        if not end_date:
            self.end_date = datetime.date.today()
            self.current_date = datetime.date.today()

        if not start_date:
            self.start_date = datetime.date(1980, 1, 1)

        self.session = get_session()
        self.box_office_links = pd.DataFrame()
        self.box_office_details = pd.DataFrame()
        self.player_box_office_details = pd.DataFrame()
        self.dates_searched_for_links = []
        self.game_links_searched = []

        if clear_data:
            self.save_data()
        self.load_data()

    def save_data(self):
        with file_lock:
            with open('{data_path}/{file_name}.pkl'.format(data_path=data_path, file_name=date_record_pickle_file_name), 'wb') as f:
                pickle.dump(self.dates_searched_for_links, f)
            with open('{data_path}/{file_name}.pkl'.format(data_path=data_path, file_name=box_score_record_pickle_file_name),
                      'wb') as f:
                    pickle.dump(self.game_links_searched, f)
            self.box_office_links.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=box_score_link_table_name), index=False, sep = '|')
            self.box_office_details.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=box_score_details_table_name), index=False, sep = '|')
            self.player_box_office_details.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=player_detail_table_name), index=False, sep = '|')

    def load_data(self):
        try:
            with file_lock:
                with open('{data_path}/{file_name}.pkl'.format(data_path=data_path,
                                                               file_name=date_record_pickle_file_name), 'rb') as f:
                    self.dates_searched_for_links = pickle.load(f)
                with open('{data_path}/{file_name}.pkl'.format(data_path=data_path,
                                                               file_name=box_score_record_pickle_file_name),
                          'rb') as f:
                    self.game_links_searched = pickle.load(f)

                self.box_office_links = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=box_score_link_table_name), sep = '|')
                self.box_office_details = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=box_score_details_table_name), sep = '|')
                self.player_box_office_details = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path, db_name=player_detail_table_name), sep = '|')
                self.box_office_details.columns = [i.replace('stat_', '') for i in self.box_office_details.columns]
        except:
            traceback.print_exc()

    def scrape_current_day_boxscore_links(self):
        for i in range(max_tries):
            try:
                padded_year = str(self.current_date.year).zfill(4)
                padded_month = str(self.current_date.month).zfill(2)
                padded_day = str(self.current_date.day).zfill(2)
                game_links = []
                url = day_scores_base_url.format(month = padded_month,
                                           day = padded_day,
                                           year = padded_year)
                print('scraping links from {}'.format(url))
                soup = get_soup(url, session = self.session)
                games = soup.find_all('p', {'class': 'links'})
                for i in games:
                    links = i.find_all('a')
                    days_games_box_score_links = [base_url + j['href'] for j in links if 'boxscores' in j['href'] and 'pbp' not in j['href'] and 'shot-chart' not in j['href']]
                    for j in days_games_box_score_links:
                        game_links.append({'box_score_url': j, 'year':padded_year, 'month':padded_month, 'day':padded_day})

                new_df = pd.DataFrame.from_dict(game_links)
                self.box_office_links = pd.concat([new_df, self.box_office_links])
                self.box_office_links = self.box_office_links.drop_duplicates()
                break
            except:
                traceback.print_exc()
                sleep_on_error()

    def scrape_box_office_details(self, url, year, month, day):
        for i in range(max_tries):
            try:
                team_data = []
                player_data = []

                soup = get_soup(url, session=self.session)
                soup = BeautifulSoup(str(soup).replace('-->', '').replace('<!--', ''), 'lxml')

                score_box = soup.find('div', {'class': 'scorebox'})
                score_box_divs = score_box.find_all('div', recursive = False)
                team_1 = score_box_divs[0]
                team_2 = score_box_divs[1]

                team_1_link = team_1.find('a', {'itemprop':'name'})['href']
                team_2_link = team_2.find('a', {'itemprop':'name'})['href']

                team_1_tag = team_1_link.split('/')[2].lower()
                team_2_tag = team_2_link.split('/')[2].lower()

                team_1_link = base_url + team_1_link
                team_2_link = base_url + team_2_link

                team_1_name = team_1.find('a', {'itemprop': 'name'}).get_text()
                team_2_name = team_2.find('a', {'itemprop': 'name'}).get_text()

                scorebox_meta = soup.find('div', {'class': 'scorebox_meta'}).find_all('div')
                location = scorebox_meta[1].get_text()

                data_tables = soup.find_all('table', {'class':'sortable stats_table'})
                # print(data_tables)
                team_1_basic_table = get_score_table(soup, team_1_tag, 'basic')
                team_1_advanced_table = get_score_table(soup, team_1_tag, 'advanced')
                team_2_basic_table = get_score_table(soup, team_2_tag, 'basic')
                team_2_advanced_table = get_score_table(soup, team_2_tag, 'advanced')

                t1_data = process_stats_tables(team_1_basic_table, team_1_advanced_table)
                t2_data = process_stats_tables(team_2_basic_table, team_2_advanced_table)

                team_1_data_self = {str(i): j for i, j in t1_data['team_data'].items()}
                t1_base_data = {
                                'team_tag':team_1_tag,
                                'team_link':team_1_link,
                                'team_name':team_1_name,
                                'opponent_tag':team_2_tag,
                                'opponent_link':team_2_link,
                                'opponent_name':team_2_name,
                                'location':location,
                                'win': 1 if float(t1_data['team_data']['pts']) > float(t2_data['team_data']['pts']) else 0,
                                'year':year,
                                'month':month,
                                'day':day
                                }

                team_2_data_self = {str(i): j for i, j in t2_data['team_data'].items()}
                t2_base_data = {
                                'team_tag':team_2_tag,
                                'team_link':team_2_link,
                                'team_name':team_2_name,
                                'opponent_tag':team_1_tag,
                                'opponent_link':team_1_link,
                                'opponent_name':team_1_name,
                                'location':location,
                                'win': 1 if float(t2_data['team_data']['pts']) > float(t1_data['team_data']['pts']) else 0,
                                'year': year,
                                'month': month,
                                'day': day
                                }

                for i in t1_data['player_data'].values():
                    t1_base_data_copy = copy.deepcopy(t1_base_data)
                    t1_base_data_copy.update(i)
                    player_data.append(t1_base_data_copy)

                for i in t2_data['player_data'].values():
                    t2_base_data_copy = copy.deepcopy(t2_base_data)
                    t2_base_data_copy.update(i)
                    player_data.append(t2_base_data_copy)

                t1_base_data.update(team_1_data_self)
                # t1_base_data.update(team_1_data_opponent)
                t2_base_data.update(team_2_data_self)
                # t2_base_data.update(team_2_data_opponent)
                team_data.append(t1_base_data)
                team_data.append(t2_base_data)

                new_df = pd.DataFrame.from_dict(team_data)
                self.box_office_details = pd.concat([new_df, self.box_office_details])
                self.box_office_details = self.box_office_details.drop_duplicates()

                new_df = pd.DataFrame.from_dict(player_data)
                self.player_box_office_details = pd.concat([new_df, self.player_box_office_details], sort = True)
                self.player_box_office_details = self.player_box_office_details.drop_duplicates()
                break
            except:
                traceback.print_exc()
                sleep_on_error()

    def scrape_date_range_boxscore_links(self, save_data = False):
        while self.current_date <= self.end_date and self.current_date >= self.start_date:
            self.current_date -= datetime.timedelta(days=1)
            if str(self.current_date) in self.dates_searched_for_links:
                continue
            self.scrape_current_day_boxscore_links()
            self.dates_searched_for_links.append(str(self.current_date))
            if save_data:
                self.save_data()

    def scrape_all_box_office_details(self, save_data = False):
        for _, i in self.box_office_links.iterrows():
            if i['box_score_url'] in self.game_links_searched:
                continue
            print('scraping game: {}'.format(i['box_score_url']))
            self.scrape_box_office_details(i['box_score_url'], i['year'], i['month'], i['day'])
            self.game_links_searched.append(i['box_score_url'])
            if save_data:
                self.save_data()

    def scrape_date_range_boxscore_links_and_details(self):
        while self.current_date <= self.end_date and self.current_date >= self.start_date:
            self.current_date -= datetime.timedelta(days=1)
            if str(self.current_date) in self.dates_searched_for_links:
                continue
            self.scrape_current_day_boxscore_links()
            self.dates_searched_for_links.append(str(self.current_date))
            self.scrape_all_box_office_details()
            self.save_data()


if __name__ == '__main__':
    scraper = Scraper(start_date = datetime.date(1980, 1, 1), end_date = datetime.date(2019, 6, 30), clear_data=False)
    scraper.scrape_date_range_boxscore_links_and_details()

