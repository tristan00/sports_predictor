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

        self.past_n_game_dataset = dict()
        self.past_n_game_dataset_combined = dict()

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
                                          'usg_pct', f'{self.feature_indicator_str}_home']
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

    # def get_dataset(self):
    #     self.load_raw_data()
    #     self.assign_home_for_teams()
    #     self.calculate_team_game_rating()
    #     self.build_past_n_game_dataset()
    #     self.combine_past_n_game_datasets()

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
    def build_past_n_game_dataset(self):
        teams = set(self.team_data['team_tag'])

        self.scaler_dict = dict()
        for i in self.initial_team_data_columns:
            scaler = QuantileTransformer()
            scaler.fit(self.team_data[i].fillna(self.team_data[i].median()).values.reshape(-1, 1))
            self.scaler_dict[i] = scaler

        self.past_n_game_dataset = dict()
        for t in tqdm.tqdm(teams):
            temp_team_data = self.team_data[self.team_data['team_tag'] == t]
            temp_team_data = temp_team_data.sort_values('date_str')

            for i in self.initial_team_data_columns:
                temp_team_data[i] = self.scaler_dict[i].transform(temp_team_data[i].values.reshape(-1, 1))

            game_ids = set(temp_team_data['game_key'])
            temp_team_data = temp_team_data.set_index('game_key')

            self.past_n_game_dataset[t] = dict()
            for g in game_ids:
                pregame_matrices = []
                for n in range(1, self.history_length + 1):
                    next_series = temp_team_data.shift(n).loc[g, self.initial_team_data_columns]
                    if next_series.empty:
                        next_series = pd.Series(data=[0 for _ in self.initial_team_data_columns],
                                                index=self.initial_team_data_columns)
                    else:
                        next_series = next_series.fillna(0)

                    pregame_matrices.append(next_series)

                if self.transpose_history_data:
                    pregame_matrix = pd.concat(pregame_matrices, axis=1).transpose().values
                else:
                    pregame_matrix = pd.concat(pregame_matrices, axis=1).values
                assert pregame_matrix.shape == (self.history_length, len(self.initial_team_data_columns))
                self.past_n_game_dataset[t][g] = pregame_matrix

        self.save_past_n_game_dataset()

    @timeit
    def save_past_n_game_dataset(self):
        with open('{data_path}/{db_name}_{history_length}_{transpose_history_data}.pkl'.format(data_path=data_path,
                                                                                               db_name=past_n_game_dataset_table_name,
                                                                                               history_length=self.history_length,
                                                                                               transpose_history_data=self.transpose_history_data),
                  'wb') as f:
            pickle.dump(self.past_n_game_dataset, f)

    @timeit
    def combine_past_n_game_datasets(self):
        all_keys = self.team_data[['game_key', 'team_tag', 'opponent_tag', 'date_str']]
        all_keys = all_keys.drop_duplicates()

        self.past_n_game_datasets_combined = dict()
        for _, row in all_keys.iterrows():
            self.past_n_game_datasets_combined.setdefault(row['team_tag'], dict())
            team_record = self.past_n_game_dataset[row['team_tag']][row['game_key']]
            opponent_record = self.past_n_game_dataset[row['opponent_tag']][row['game_key']]
            diff = team_record - opponent_record
            self.past_n_game_datasets_combined[row['team_tag']][row['game_key']] = np.hstack([team_record,
                                                                                              opponent_record,
                                                                                              diff])
        self.save_past_n_game_dataset_combined()

    @timeit
    def save_past_n_game_dataset_combined(self):
        with open('{data_path}/{db_name}.pkl'.format(data_path=data_path,
                                                     db_name=past_n_game_dataset_table_name), 'wb') as f:
            pickle.dump(self.past_n_game_datasets_combined, f)

    @timeit
    def load_past_n_game_dataset_combined(self):
        with open('{data_path}/{db_name}.pkl'.format(data_path=data_path,
                                                     db_name=past_n_game_dataset_table_name), 'rb') as f:
            self.past_n_game_datasets_combined = pickle.load(f)

    @timeit
    def get_labeled_data(self):
        self.load_processed_data()
        self.load_past_n_game_dataset_combined()

        all_keys = self.team_data[['game_key', 'team_tag', 'opponent_tag', 'date_str', self.target]]
        all_keys = all_keys.drop_duplicates()

        x = []
        y = []
        for _, row in all_keys.iterrows():
            y.append(row[self.target])
            x.append(self.past_n_game_datasets_combined[row['team_tag']][row['game_key']])

        return np.array(x), np.array(y)

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
        self.team_data = self.team_data.sort_values('date_str').copy()
        new_col = f'{self.feature_indicator_str}_{self.team_str}_{self.pregame_rating_str}_{rating_type}'
        self.initial_team_data_columns.append(new_col)

        self.team_data[new_col] = None
        self.team_data[f'{self.feature_indicator_str}_{self.team_str}_{self.postgame_rating_str}_{rating_type}'] = None

        for i, r in tqdm.tqdm(self.team_data.iterrows()):
            team_previous_record = self.get_most_recent_team_record_before_date(self.team_data, r['team_tag'],
                                                                                r['date_str'])
            opponent_previous_record = self.get_most_recent_team_record_before_date(self.team_data, r['opponent_tag'],
                                                                                    r['date_str'])

            if team_previous_record.empty:
                team_previous_rating = starting_rating
            else:
                team_previous_rating = team_previous_record[
                    f'{self.feature_indicator_str}_{self.team_str}_{self.postgame_rating_str}_{rating_type}']

            if opponent_previous_record.empty:
                opponent_previous_rating = starting_rating
            else:
                opponent_previous_rating = opponent_previous_record[
                    f'{self.feature_indicator_str}_{self.team_str}_{self.postgame_rating_str}_{rating_type}']
            self.team_data.loc[i, new_col] = team_previous_rating
            self.team_data.loc[
                i, f'{self.feature_indicator_str}_{self.team_str}_{self.postgame_rating_str}_{rating_type}'] = get_new_rating(
                team_previous_rating,
                opponent_previous_rating,
                r['win'], multiplier=1, rating_type=rating_type)
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
