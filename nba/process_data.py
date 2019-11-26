import pandas as pd
import numpy as np
from nba.common import (
    data_path,
    box_score_link_table_name,
    box_score_details_table_name,
    player_detail_table_name,
    starting_rating,
    get_new_rating,
    file_lock,
    processed_team_data_table_name,
    timeit,
    processed_player_data_table_name,
    combined_feature_file_data_table_name,
    past_n_game_dataset_table_name,
    past_n_game_dataset_combined_table_name
)
import copy
import pickle
from sklearn.decomposition import PCA
# import multiprocessing
from collections import Counter
import tqdm
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer


@timeit
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

    def __init__(self, encoder_size=64, history_length=16, transpose_history_data=True):
        self.encoder_size = encoder_size
        self.history_length = history_length
        self.transpose_history_data = transpose_history_data

        self.cache = dict()

        self.feature_indicator_str = 'feature'
        self.team_str = 'team'
        self.player_str = 'player'
        self.opponent_str = 'opponent'
        self.pregame_rating_str = 'pregame_rating'
        self.postgame_rating_str = 'postgame_rating'

        self.initial_team_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                          'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                          'ft_pct',
                                          'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                          'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                          'ts_pct',
                                          'usg_pct', f'{self.feature_indicator_str}_home',
                                          f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_0',
                                          'win']
        self.initial_player_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                            'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                            'ft_pct',
                                            'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                            'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                            'ts_pct',
                                            'usg_pct']
        self.target = 'win'
        self.id_columns = ['team_tag', 'team_link', 'team_name', 'opponent_tag', 'opponent_name', 'opponent_link',
                           'location',
                           'date_str', 'game_key', 'team_game_key', 'player_link']

    def update_raw_datasets(self):
        self.load_raw_data()
        self.assign_home_for_teams()
        self.calculate_team_game_rating(0)
        # self.calculate_team_game_rating(1)
        # self.calculate_team_game_rating(2)
        # self.calculate_team_game_rating(3)
        self.save_processed_data()


    def load_raw_data(self):
        self.team_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                        db_name=box_score_details_table_name),
                                     sep='|', low_memory=False)
        self.player_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                          db_name=player_detail_table_name), sep='|',
                                       low_memory=False)

        self.team_dfs_dict = dict()
        self.team_dfs_dict = dict()

        self.team_features = pd.DataFrame()
        self.player_features = pd.DataFrame()

        self.team_data['date_str'] = self.team_data.apply(
            lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis=1)
        self.team_data['game_key'] = self.team_data.apply(
            lambda x: str(sorted([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])])), axis=1)
        self.team_data['team_game_key'] = self.team_data.apply(
            lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

        self.player_data['date_str'] = self.player_data.apply(
            lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis=1)
        self.player_data['game_key'] = self.player_data.apply(
            lambda x: str(sorted([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])])), axis=1)
        self.player_data['team_game_key'] = self.player_data.apply(
            lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

        self.team_data = self.team_data.sort_values('date_str')
        self.player_data = self.player_data.sort_values('date_str')
        self.save_processed_data()

    def save_processed_data(self):
        self.team_data.to_csv('{data_path}/{db_name}_processed.csv'.format(data_path=data_path,
                                                                           db_name=box_score_details_table_name),
                              sep='|', index=False)
        self.player_data.to_csv('{data_path}/{db_name}_processed.csv'.format(data_path=data_path,
                                                                             db_name=player_detail_table_name), sep='|',
                                index=False)

    def load_processed_data(self):
        self.team_data = pd.read_csv('{data_path}/{db_name}_processed.csv'.format(data_path=data_path,
                                                                                  db_name=box_score_details_table_name),
                                     sep='|', low_memory=False)
        self.player_data = pd.read_csv('{data_path}/{db_name}_processed.csv'.format(data_path=data_path,
                                                                                    db_name=player_detail_table_name),
                                       sep='|', low_memory=False)

    @timeit
    def build_timeseries(self, history_length, transpose_history_data):
        print(f'build_timeseries: {history_length} {transpose_history_data}')
        self.load_processed_data()
        teams = set(self.team_data['team_tag'])

        self.scaler_dict = dict()
        for i in self.initial_team_data_columns:
            scaler = StandardScaler()
            scaler.fit_transform(self.team_data[i].fillna(self.team_data[i].median()).values.reshape(-1, 1))
            self.scaler_dict[i] = scaler
            team_data_opponent = self.team_data.copy()

        team_data_opponent['temp_column'] = team_data_opponent['team_tag']
        team_data_opponent['team_tag'] = team_data_opponent['opponent_tag']
        team_data_opponent['opponent_tag'] = team_data_opponent['temp_column']
        team_data_opponent['game_key'] = team_data_opponent.apply(
            lambda x: str(sorted([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])])), axis=1)
        team_data_opponent['team_game_key'] = team_data_opponent.apply(
            lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

        opponent_columns = list()
        for i in self.initial_team_data_columns:
            team_data_opponent['{}_opponent'.format(i)] = team_data_opponent[i]
            opponent_columns.append('{}_opponent'.format(i))

        self.team_data = self.team_data.merge(team_data_opponent[['team_tag', 'opponent_tag', 'game_key'] + opponent_columns])
        past_n_game_dataset = dict()
        temp_team_df_dict = dict()
        for t in tqdm.tqdm(teams):
            temp_team_df_dict[t] = self.team_data[self.team_data['team_tag'] == t]

        combined_columns = self.initial_team_data_columns + opponent_columns

        for t in tqdm.tqdm(teams):
            past_n_game_dataset[t] = dict()
            temp_team_df_dict[t] = temp_team_df_dict[t].sort_values('date_str')

            game_ids = set(temp_team_df_dict[t]['game_key'])
            temp_team_df_dict[t] = temp_team_df_dict[t].set_index('game_key')

            for g in game_ids:
                g_iloc = temp_team_df_dict[t].index.get_loc(g)
                pregame_matrix = temp_team_df_dict[t].shift(history_length).iloc[g_iloc:g_iloc + history_length][combined_columns].fillna(0).values

                while pregame_matrix.shape[0] < history_length:
                    new_array = np.array([[0 for _ in combined_columns]])
                    pregame_matrix = np.vstack([new_array, pregame_matrix])

                diff = pregame_matrix[:,0:len(self.initial_team_data_columns)] - pregame_matrix[:,len(self.initial_team_data_columns):]
                pregame_matrix = np.hstack([pregame_matrix, diff])

                if transpose_history_data:
                    pregame_matrix = pregame_matrix.transpose()

                past_n_game_dataset[t][g] = pregame_matrix

        self.save_past_n_game_dataset(past_n_game_dataset, history_length, transpose_history_data)

    @timeit
    def save_past_n_game_dataset(self, past_n_game_dataset, history_length, transpose_history_data):
        with open(f'{data_path}/{past_n_game_dataset_table_name}_{history_length}_{transpose_history_data}.pkl',
                  'wb') as f:
            pickle.dump(past_n_game_dataset, f)

    @timeit
    def load_past_n_game_dataset(self, history_length, transpose_history_data):
        with open(f'{data_path}/{past_n_game_dataset_table_name}_{history_length}_{transpose_history_data}.pkl',
                  'rb') as f:
            return pickle.load(f)

    @timeit
    def combine_timeseries(self, history_length, transpose_history_data):
        self.load_processed_data()
        all_keys = self.team_data[['game_key', 'team_tag', 'opponent_tag', 'date_str']]
        all_keys = all_keys.drop_duplicates()
        past_n_game_dataset = self.load_past_n_game_dataset(history_length, transpose_history_data)

        past_n_game_datasets_combined  = dict()
        for _, row in all_keys.iterrows():
            past_n_game_datasets_combined.setdefault(row['team_tag'], dict())
            team_record = past_n_game_dataset[row['team_tag']][row['game_key']]
            opponent_record = past_n_game_dataset[row['opponent_tag']][row['game_key']]
            diff = team_record - opponent_record
            past_n_game_datasets_combined[row['team_tag']][row['game_key']] = np.hstack([team_record,
                                                                                              opponent_record,
                                                                                              diff])
        self.save_past_n_game_dataset_combined(past_n_game_datasets_combined, history_length, transpose_history_data)

    @timeit
    def save_past_n_game_dataset_combined(self, past_n_game_datasets_combined, history_length, transpose_history_data):
        with open(f'{data_path}/{past_n_game_dataset_combined_table_name}_{history_length}_{transpose_history_data}.pkl', 'wb') as f:
            pickle.dump(past_n_game_datasets_combined, f)

    @timeit
    def load_past_n_game_dataset_combined(self, history_length, transpose_history_data):
        with open(f'{data_path}/{past_n_game_dataset_combined_table_name}_{history_length}_{transpose_history_data}.pkl', 'rb') as f:
             return pickle.load(f)

    @timeit
    def get_labeled_data(self, history_length, transpose_history_data):
        self.load_processed_data()
        past_n_game_dataset_combined = self.load_past_n_game_dataset_combined(history_length, transpose_history_data)
        x2_cols = [f'{self.feature_indicator_str}_home', f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_0']

        all_keys = self.team_data[['game_key', 'team_tag', 'opponent_tag', 'date_str', self.target] + x2_cols]
        all_keys = all_keys.drop_duplicates()

        x1 = []
        x2 = []
        y = []
        for _, row in all_keys.iterrows():
            y.append(row[self.target])
            x1.append(past_n_game_dataset_combined[row['team_tag']][row['game_key']])
            x2.append([row[f'{self.feature_indicator_str}_home'],
                       row[f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_0']/1000.0])

        return np.array(x1), np.array(x2), np.array(y)

    @timeit
    def assign_home_for_teams(self):
        self.load_processed_data()
        home_dict = find_team_home_loc(self.team_data)
        self.team_data[f'{self.feature_indicator_str}_home'] = self.team_data.apply(
            lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis=1)
        self.save_processed_data()

    @timeit
    def calculate_team_game_rating(self, rating_type):
        self.load_processed_data()
        team_data_copy = self.team_data.sort_values('date_str').copy()
        new_col_pre = f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_{rating_type}'
        new_col_post = f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_{rating_type}'
        # self.initial_team_data_columns.append(new_col)

        team_data_copy = team_data_copy[['team_tag', 'opponent_tag', 'date_str']]
        team_data_copy[new_col_pre] = None
        team_data_copy[new_col_post] = None

        for i, r in tqdm.tqdm(self.team_data.iterrows()):
            team_previous_record = self.get_most_recent_team_record_before_date(team_data_copy, r['team_tag'],
                                                                                r['date_str'])
            opponent_previous_record = self.get_most_recent_team_record_before_date(team_data_copy, r['opponent_tag'],
                                                                                    r['date_str'])

            if team_previous_record.empty:
                team_previous_rating = starting_rating
            else:
                team_previous_rating = team_previous_record[new_col_post]

            if opponent_previous_record.empty:
                opponent_previous_rating = starting_rating
            else:
                opponent_previous_rating = opponent_previous_record[new_col_post]
            team_data_copy.loc[i, new_col_pre] = team_previous_rating
            team_data_copy.loc[i, new_col_post] = get_new_rating(
                team_previous_rating,
                opponent_previous_rating,
                r['win'], multiplier=1, rating_type=rating_type)

        self.team_data = self.team_data.merge(team_data_copy)
        self.save_processed_data()

    #################################################################################################################
    # Helper methods

    def presplit_teams_and_players(self):
        for team_tag in set(self.team_data['team_tag']):
            self.team_dfs_dict[team_tag] = self.team_data[self.team_data['team_tag']]
            self.team_dfs_dict[team_tag] = self.team_dfs_dict[team_tag].sort_values('date_str')

        for player_link in set(self.player_data['player_link']):
            self.player_data[player_link] = self.player_data[self.player_data['player_link']]
            self.player_data[player_link] = self.player_data[player_link].sort_values('date_str')

    def get_most_recent_team_record_before_date(self, df, tag, date_str):
        sub_df = df[(df['date_str'] < date_str) & (df['team_tag'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

    def get_team_record_right_after_date(self, df, tag, date_str):
        sub_df = df[(df['date_str'] > date_str) & (df['team_tag'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

    def get_most_recent_player_record_before_date(self, df, tag, date_str):
        sub_df = df[(df['date_str'] < date_str) & (df['player_link'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

    def get_player_record_right_after_date(self, df, tag, date_str):
        sub_df = df[(df['date_str'] > date_str) & (df['player_link'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df


def create_data_files():
    dm = DataManager()
    dm.load_raw_data()
    dm.assign_home_for_teams()
    dm.build_past_n_game_dataset()
    dm.combine_past_n_game_datasets()


if __name__ == '__main__':
    create_data_files()
