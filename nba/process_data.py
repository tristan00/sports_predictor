import pandas as pd
import numpy as np
from nba.common import (
                    data_path,
                    box_score_link_table_name,
                    box_score_details_table_name,
                    player_detail_table_name,
                    starting_rating,
                    get_new_rating,
                    file_lock
                    )
# import multiprocessing
from collections import Counter
import tqdm

def find_team_home_loc(df):
    home_dict = dict()
    team_tags = set(df['team_tag'])

    for i in team_tags:
        team_data = df[df['team_tag'] == i]
        team_years = set(df['year'])

        for y in team_years:
            team_year_data = team_data[team_data['year'] == y]

            c = Counter()
            for _, j in team_year_data.iterrows():
                c[j['location']] += 1
            if len(c) > 0:
                home_dict[(i, y)] = c.most_common(1)[0][0]
    return home_dict

class DataManager():
    feature_indicator_str = 'feature'
    team_str = 'team'
    opponent_str = 'opponent'
    pregame_rating_str = 'pregame_rating'
    postgame_rating_str = 'postgame_rating'

    initial_data_columns = ['stat_ast', 'stat_ast_pct','stat_blk', 'stat_blk_pct','stat_def_rtg','stat_drb','stat_drb_pct','stat_efg_pct','stat_fg',
                  'stat_fg3','stat_fg3_pct','stat_fg3a','stat_fg3a_per_fga_pct','stat_fg_pct','stat_fga','stat_ft','stat_ft_pct',
                  'stat_fta','stat_fta_per_fga_pct','stat_mp','stat_off_rtg','stat_orb','stat_orb_pct','stat_pf','stat_plus_minus',
                  'stat_pts','stat_stl','stat_stl_pct','stat_tov', 'stat_tov_pct', 'stat_trb','stat_trb_pct', 'stat_ts_pct',
                  'stat_usg_pct','win']

    def __init__(self, team_data, player_data):
        self.team_data = team_data
        self.player_data = player_data

        self.team_dfs_dict = dict()
        self.team_dfs_dict = dict()

        self.team_features = pd.DataFrame()
        self.player_features = pd.DataFrame()

        self.team_data['date_str'] = self.team_data.apply(lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis = 1)
        self.team_data['game_key'] = self.team_data.apply(lambda x:  str(sorted([x['date_str'], x['team_tag'], x['opponent_tag']])), axis = 1)
        self.team_data['team_game_key'] = self.team_data.apply(lambda x:  str([x['date_str'], x['team_tag'], x['opponent_tag']]), axis = 1)

        self.player_data['date_str'] = self.player_data.apply(lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis = 1)
        self.player_data['game_key'] = self.player_data.apply(lambda x:  str(sorted([x['date_str'], x['team_tag'], x['opponent_tag']])), axis = 1)
        self.player_data['team_game_key'] = self.player_data.apply(lambda x:  str([x['date_str'], x['team_tag'], x['opponent_tag']]), axis = 1)

    def create_team_features(self):
        self.assign_home_for_teams()
        for i in [1, 3, 5, 10, 25, 50, 100]:
            self.calculate_moving_averages(i)
        for i in [0, 1, 2, 3]:
            self.calculate_team_game_rating(i)

    def assign_home_for_teams(self):
        home_dict = find_team_home_loc(self.team_data)
        self.team_data[f'{self.feature_indicator_str}_home'] = self.team_data.apply(lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis = 1)

    def calculate_moving_averages(self, n):
        teams = set(self.team_data['team_tag'])
        for t in teams:
            for c in self.initial_data_columns:
                self.team_data.loc[self.team_data['team_tag'] == t, f'{self.feature_indicator_str }_{self.team_str}_rl_avg_{c}_{n}'] = self.team_data[self.team_data['team_tag'] == t].shift(periods=1).rolling(window=n)[c].mean()

    def calculate_team_game_rating(self, rating_type):
        self.team_data = self.team_data.sort_values('date_str')
        self.team_data[f'{self.feature_indicator_str }_{self.team_str}_{self.pregame_rating_str}_{rating_type}'] = None
        self.team_data[f'{self.feature_indicator_str }_{self.team_str}_{self.postgame_rating_str}_{rating_type}'] = None

        for i, r in tqdm.tqdm(self.team_data.iterrows()):
            team_previous_record = self.get_most_recent_team_record_before_date(r['team_tag'], r['date_str'])
            opponent_previous_record = self.get_most_recent_team_record_before_date(r['opponent_tag'], r['date_str'])

            if team_previous_record.empty:
                team_previous_rating = starting_rating
            else:
                team_previous_rating = team_previous_record[f'{self.feature_indicator_str }_{self.team_str}_{self.postgame_rating_str}_{rating_type}']

            if opponent_previous_record.empty:
                opponent_previous_rating = starting_rating
            else:
                opponent_previous_rating = opponent_previous_record[f'{self.feature_indicator_str }_{self.team_str}_{self.postgame_rating_str}_{rating_type}']
            self.team_data.loc[i, f'{self.feature_indicator_str }_{self.team_str}_{self.pregame_rating_str}_{rating_type}'] = team_previous_rating
            self.team_data.loc[i, f'{self.feature_indicator_str }_{self.team_str}_{self.postgame_rating_str}_{rating_type}'] = get_new_rating(team_previous_rating,
                                                                                                                                   opponent_previous_rating,
                                                                                                                                   r['win'], multiplier = 1, rating_type = 0)

    #################################################################################################################
    # Helper methods

    def presplit_teams_and_players(self):
        for team_tag in set(self.team_data['team_tag']):
            self.team_dfs_dict[team_tag] = self.team_data[self.team_data['team_tag']]
            self.team_dfs_dict[team_tag] = self.team_dfs_dict[team_tag].sort_values('date_str')

        for player_link in set(self.player_data['player_link']):
            self.player_data[player_link] = self.player_data[self.player_data['player_link']]
            self.player_data[player_link] = self.player_data[player_link].sort_values('date_str')

    def get_most_recent_team_record_before_date(self, tag, date_str):
        sub_df = self.team_data[(self.team_data['date_str'] < date_str) & (self.team_data['team_tag'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

    def get_team_record_right_after_date(self,tag, date_str):
        sub_df = self.team_data[(self.team_data['date_str'] > date_str) & (self.team_data['team_tag'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

def create_data_files():
    with file_lock:
        team_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                   db_name=box_score_details_table_name),
                                sep='|', low_memory=False)
        player_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=player_detail_table_name), sep='|', low_memory=False)

    dm = DataManager(team_data, player_data)
    dm.create_team_features()
    df = dm.team_data.sort_values('feature_team_postgame_rating_1')
    df.describe()

if __name__ == '__main__':
    create_data_files()

