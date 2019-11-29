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
    past_n_game_dataset_combined_table_name,
parse_minutes_played
)
from keras import layers, models, callbacks, optimizers
import json
import copy
import pickle
from sklearn.decomposition import PCA
# import multiprocessing
from collections import Counter
import tqdm
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import collections
from sklearn.model_selection import train_test_split
import gc


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

    max_players_to_consider_per_team = 12

    def __init__(self, encoder_size=64,
                 history_length=16,
                 testing=0,
                 fill_nans=True,
                 data_scaling=None):
        self.data_scaling = data_scaling
        self.fill_nans = fill_nans
        self.encoder_size = encoder_size
        self.history_length = history_length

        self.cache = dict()

        self.feature_indicator_str = 'feature'
        self.team_str = 'team'
        self.player_str = 'player'
        self.opponent_str = 'opponent'
        self.pregame_rating_str = 'pregame_rating'
        self.postgame_rating_str = 'postgame_rating'

        self.initial_team_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                          'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                          'ft_pct', 'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                          'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                          'ts_pct', 'usg_pct']
        self.initial_player_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                            'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                            'ft_pct',
                                            'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                            'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                            'ts_pct',
                                            'usg_pct']
        self.target = 'win'
        self.id_columns = ['team_tag', 'team_link', 'team_name', 'opponent_tag', 'opponent_name', 'opponent_link',
                           'location', 'date_str', 'game_key', 'team_game_key', 'player_link']

        self.key_columns = ['game_key', 'team_tag', 'opponent_tag', 'date_str', 'team_game_key']
        self.standalone_feature_columns = []
        self.diff_feature_cols = []

        self.testing = testing

    @timeit
    def update_raw_datasets(self):
        self.load_raw_data()
        self.process_minutes_played()
        self.create_feature_target_df()
        self.calculate_team_game_rating(0)
        # self.calculate_team_game_rating(1)
        # self.calculate_team_game_rating(2)
        # self.calculate_team_game_rating(3)
        self.encode_dates()
        print(1, self.feature_df.shape)
        self.assign_date_since_last_game()
        print(2, self.feature_df.shape)
        self.assign_home_for_teams()
        print(3, self.feature_df.shape)
        self.build_moving_average_features(5)
        print(4, self.feature_df.shape)
        # self.build_moving_average_features(10)
        # print(5, self.feature_df.shape)
        # self.build_moving_average_features(25)
        self.build_moving_average_features(100)
        print(6, self.feature_df.shape)
        self.build_event_features()
        print(7, self.feature_df.shape)

        if self.data_scaling:
            self.scale_data()

        print(dict(self.team_data['team_game_key'].value_counts()))
        self.save_processed_data()
        self.save_columns()
        self.save_feature_df()

    @timeit
    def get_labeled_data(self, history_length, get_history_data, team_encoding_size,
                                    team_encoding_type,
                         player_encoding_1_size,
                                player_encoding_2_size,
                         player_encoding_type):
        self.load_processed_data()
        self.load_feature_df()
        self.load_columns()
        if get_history_data:
            past_n_game_dataset_combined = self.load_past_n_game_dataset_combined(history_length,
                                                                                  team_encoding_size = team_encoding_size,
                                                                                  team_encoding_type = team_encoding_type,
                                                                                  player_encoding_1_size = player_encoding_1_size,
                                                                                  player_encoding_2_size=player_encoding_2_size,
                                                                                  player_encoding_type = player_encoding_type,)
        all_features = self.standalone_feature_columns.copy()

        for i in self.diff_feature_cols:
            all_features.append('{}_diff'.format(i))

        y, x1, x2 = [], [], []
        for _, row in self.feature_df.iterrows():
            if row[self.target] == 0:
                y.append([1, 0])
            elif row[self.target] == 1:
                y.append([0, 1])
            else:
                raise Exception('invalid target: {}'.format(dict(row)))
            if self.fillna:
                row = row.fillna(0)
            if get_history_data:
                x1.append(past_n_game_dataset_combined[row['team_tag']][row['game_key']])
            x2.append(row[all_features])

        return np.array(x1), np.array(x2), np.array(y), all_features


    @timeit
    def load_raw_data(self):
        if self.testing:
            self.team_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                            db_name=box_score_details_table_name),
                                                                            sep='|', low_memory=False,
                                                                            nrows=self.testing)
            self.player_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                              db_name=player_detail_table_name),
                                                                              sep='|', low_memory=False,
                                                                              nrows=self.testing)
        else:
            self.team_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                            db_name=box_score_details_table_name),
                                                                            sep='|', low_memory=False)
            self.player_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                              db_name=player_detail_table_name),
                                                                              sep='|', low_memory=False)

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
        self.player_data['player_game_key'] = self.player_data.apply(
            lambda x: str([str(x['player_link']), str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

        self.team_data = self.team_data.sort_values('date_str')
        self.player_data = self.player_data.sort_values('date_str')
        self.save_processed_data()

    #################################################################################################################
    # Time series creation

    @timeit
    def build_team_timeseries(self, history_length, team_encoding_size, team_encoding_type):
        teams = set(self.team_data['team_tag'])

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

        team_data_combined_with_opponent = self.team_data.merge(
            team_data_opponent[['team_tag', 'opponent_tag', 'game_key'] + opponent_columns])

        for i in self.initial_team_data_columns:
            team_data_combined_with_opponent['{}_diff'.format(i)] = team_data_combined_with_opponent[i] - team_data_combined_with_opponent['{}_opponent'.format(i)]

        past_n_game_dataset = dict()
        temp_team_df_dict = dict()


        combined_columns = self.initial_team_data_columns + opponent_columns

        encoder_1 = Encoder(team_encoding_type, team_encoding_size, 'player_encoding_1', None)
        try:
            encoder_1.load()
        except:
            encoder_1.fit(team_data_combined_with_opponent[combined_columns].values)

        preds = encoder_1.transform(team_data_combined_with_opponent[combined_columns].values)
        encoded_cols = [i for i in range(encoder_1.encoder_dims)]
        team_data_combined_with_opponent = team_data_combined_with_opponent.set_index(['team_tag', 'date_str', 'game_key'])
        team_data_combined_with_opponent = pd.DataFrame(data = preds,
                                        index = team_data_combined_with_opponent.index,
                                                        columns=encoded_cols)
        team_data_combined_with_opponent = team_data_combined_with_opponent.reset_index()


        for t in tqdm.tqdm(teams):
            team_data_copy = team_data_combined_with_opponent[team_data_combined_with_opponent['team_tag'] == t].copy()
            team_data_copy = team_data_copy.sort_values('date_str')
            temp_team_df_dict[t] = team_data_copy

        np_arrays = []
        for t in tqdm.tqdm(teams):
            past_n_game_dataset[t] = dict()
            temp_team_df_dict[t] = temp_team_df_dict[t].sort_values('date_str')

            game_ids = set(temp_team_df_dict[t]['game_key'])
            temp_team_df_dict[t] = temp_team_df_dict[t].set_index('game_key')

            for g in game_ids:
                g_iloc = temp_team_df_dict[t].index.get_loc(g)
                pregame_matrix = temp_team_df_dict[t].iloc[g_iloc-history_length:g_iloc][encoded_cols].fillna(0).values

                while pregame_matrix.shape[0] < history_length:
                    new_array = np.array([[0 for _ in encoded_cols]])
                    pregame_matrix = np.vstack([new_array, pregame_matrix])

                past_n_game_dataset[t][g] = pregame_matrix

                if team_encoding_type:
                    np_arrays.append(pregame_matrix)

        return past_n_game_dataset

    @timeit
    def build_player_timeseries(self, history_length, player_encoding_1_size, player_encoding_2_size, player_encoding_type):

        player_data_copy = self.player_data.copy()
        players = list(set(self.player_data['player_link']))
        player_data_copy['opponent_team_game_key'] = player_data_copy.apply(lambda x: str([str(x['date_str']), str(x['opponent_tag']), str(x['team_tag'])]), axis=1)

        print('build_player_timeseries 1')
        for t in tqdm.tqdm(players):
            player_data_copy.loc[player_data_copy['player_link'] == t, 'mp_trailing_avg'] = \
                player_data_copy[player_data_copy['player_link'] == t].shift(periods=1).rolling(window=history_length)['mp'].mean()

        team_game_ids = sorted(list(set(self.player_data['team_game_key'])))
        team_game_ids_vc = self.player_data['team_game_key'].value_counts()
        player_data_copy = player_data_copy.sort_values(['team_game_key', 'date_str'])

        key_opponent_mapping = dict()
        counter = 0
        player_ordering_dict = dict()
        player_result_dict = dict()
        result_dict = dict()
        result_opponent_dict = dict()
        player_team_game_id_dict = dict()

        print('build_player_timeseries 2')
        for i in tqdm.tqdm(team_game_ids):
            sub_df = player_data_copy.iloc[counter:counter+team_game_ids_vc[i]]
            counter += team_game_ids_vc[i]
            sub_df = sub_df.copy()
            assert sub_df.shape[0] == team_game_ids_vc[i] and len(set(sub_df['team_game_key'])) == 1 and sub_df['team_game_key'].tolist()[0] == i

            key_opponent_mapping[i] = sub_df.iloc[0]['opponent_team_game_key']
            sub_df = sub_df.sort_values('mp_trailing_avg', ascending = False)
            team_game_id_player_order = sub_df['player_link'].tolist()[:self.max_players_to_consider_per_team]
            player_ordering_dict[i] = team_game_id_player_order
            for j in team_game_id_player_order:
                player_team_game_id_dict.setdefault(j, set())
                player_team_game_id_dict[j].add(i)



        print('build_player_timeseries 3, build first encoder')

        print(dict(player_data_copy[self.initial_player_data_columns].isna().sum()/player_data_copy.shape[0]))

        encoder_1 = Encoder(player_encoding_type, player_encoding_1_size, 'player_encoding_1', None)
        try:
            encoder_1.load()
        except:
            encoder_1.fit(player_data_copy[self.initial_player_data_columns].values)
        preds = encoder_1.transform(player_data_copy[self.initial_player_data_columns])
        player_data_copy = player_data_copy.set_index(['player_link', 'date_str', 'team_game_key'])
        encoded_cols = [i for i in range(encoder_1.encoder_dims)]
        player_data_copy = pd.DataFrame(data = preds,
                                        index = player_data_copy.index,
                                        columns = encoded_cols)
        player_data_copy = player_data_copy.reset_index()


        print('build_player_timeseries 4')

        player_team_game_ids = sorted(list(set(player_team_game_id_dict.keys())))
        player_data_copy = player_data_copy.sort_values(['player_link', 'date_str'])

        player_result_dict_key = collections.namedtuple('player_result_dict_key', 'player_id team_game_id')
        for p in tqdm.tqdm(player_team_game_ids):

            player_data_copy_temp = player_data_copy[player_data_copy['player_link'] == p]
            # player_data_copy_temp = player_data_copy.iloc[counter:counter + player_team_game_ids_vc[p]]
            player_data_copy_temp = player_data_copy_temp.set_index('team_game_key')

            # print(p, set(player_data_copy_temp['player_link']))
            # print(player_data_copy_temp.shape[0], player_team_game_ids_vc[p])
            # print(dict(player_data_copy_temp['player_link'].value_counts()))
            # assert player_data_copy_temp.shape[0] == player_team_game_ids_vc[p] and len(set(player_data_copy_temp['player_link'])) == 1 and player_data_copy_temp['player_link'].tolist()[0] == p

            for g in player_team_game_id_dict.get(p, set()):
                g_iloc = player_data_copy_temp.index.get_loc(g)
                pregame_matrix = player_data_copy_temp.iloc[g_iloc-history_length:g_iloc][encoded_cols].fillna(0).values

                while pregame_matrix.shape[0] < history_length:
                    new_array = np.array([[0 for _ in range(encoder_1.encoder_dims)]])

                    # print(pregame_matrix.shape, new_array.shape)
                    pregame_matrix = np.vstack([new_array, pregame_matrix])
                player_result_dict[player_result_dict_key(player_id=p, team_game_id=g)] = pregame_matrix

        print('build_player_timeseries 5')
        for g in tqdm.tqdm(team_game_ids):
            np_arrays = list()

            for i in range(self.max_players_to_consider_per_team):
                if i >= len(player_ordering_dict[g]):
                    pregame_matrix = np.zeros((history_length, encoder_1.encoder_dims))
                    np_arrays.append(pregame_matrix)
                else:
                    p = player_ordering_dict[g][i]
                    np_arrays.append(player_result_dict[player_result_dict_key(player_id=p, team_game_id=g)])

            result_dict[g] = np.hstack(np_arrays)

        print('build_player_timeseries 6')
        for i in tqdm.tqdm(team_game_ids):
            v1 = result_dict[i]
            v2 = result_dict[key_opponent_mapping[i]]
            diff = v1 - v2
            result_opponent_dict[i] = np.hstack([v1, v2, diff])

        del result_dict, player_result_dict, player_ordering_dict, player_data_copy
        gc.collect()

        # print('build_player_timeseries 7')
        # if player_encoding_type:
        #     np_arrays = np.vstack(list(result_opponent_dict.values()))
        #     encoder_2 = Encoder(player_encoding_type, player_encoding_2_size, 'player_encoding_2', None)
        #     try:
        #         encoder_2.load()
        #     except:
        #         encoder_2.fit(np_arrays)
        #     for i in tqdm.tqdm(result_opponent_dict.keys()):
        #         result_opponent_dict[i] = encoder_2.transform(result_opponent_dict[i])
        return result_opponent_dict

    @timeit
    def combine_timeseries(self, past_n_game_team_dataset,
                                past_n_game_player_dataset,
                                history_length,
                                team_encoding_size,
                                team_encoding_type,
                                player_encoding_1_size,
                                player_encoding_2_size,
                                player_encoding_type):
        print(f'combine_timeseries: {history_length}, {team_encoding_size}, {team_encoding_type}, {player_encoding_1_size}, {player_encoding_2_size}, {player_encoding_type}')
        self.load_processed_data()
        self.load_columns()
        all_keys = self.team_data[['game_key', 'team_tag', 'opponent_tag', 'date_str', 'team_game_key']]
        all_keys = all_keys.drop_duplicates()
        # past_n_game_dataset = self.load_past_n_game_dataset(history_length, transpose_history_data)

        past_n_game_datasets_combined = dict()
        for _, row in all_keys.iterrows():
            past_n_game_datasets_combined.setdefault(row['team_tag'], dict())
            team_record = past_n_game_team_dataset[row['team_tag']][row['game_key']]
            opponent_record = past_n_game_team_dataset[row['opponent_tag']][row['game_key']]
            player_record = past_n_game_player_dataset[row['team_game_key']]
            diff = team_record - opponent_record
            past_n_game_datasets_combined[row['team_tag']][row['game_key']] = np.hstack([team_record,
                                                                                              opponent_record,
                                                                                             diff,
                                                                                             player_record])


        self.save_past_n_game_dataset_combined(past_n_game_datasets_combined,
                                               history_length,
                                team_encoding_size,
                                team_encoding_type,
                                player_encoding_1_size,
                                player_encoding_2_size,
                                player_encoding_type)



    @timeit
    def build_timeseries(self, history_length, team_encoding_size, team_encoding_type, player_encoding_1_size,
                                player_encoding_2_size,
                                player_encoding_type):
        print(f'build_timeseries: {history_length}, {team_encoding_size}, {team_encoding_type}, {player_encoding_1_size}, {player_encoding_2_size}, {player_encoding_type}')
        self.load_processed_data()
        self.load_columns()
        past_n_game_player_dataset = self.build_player_timeseries(history_length, player_encoding_1_size, player_encoding_2_size, player_encoding_type)
        past_n_game_team_dataset = self.build_team_timeseries(history_length, team_encoding_size, team_encoding_type)


        # self.save_past_n_game_dataset(past_n_game_dataset, history_length, transpose_history_data)
        self.combine_timeseries(past_n_game_team_dataset,
                                past_n_game_player_dataset,
                                history_length,
                                team_encoding_size,
                                team_encoding_type,
                                player_encoding_1_size,
                                player_encoding_2_size,
                                player_encoding_type)

    #################################################################################################################
    # Feature creation
    @timeit
    def create_feature_target_df(self):
        self.feature_df = self.team_data[['team_game_key', 'game_key', 'team_tag', 'opponent_tag', self.target]].copy()
        self.feature_df = self.feature_df.set_index('team_game_key')

    @timeit
    def build_moving_average_features(self, n):
        team_features = self.team_data.copy()
        teams = set(team_features['team_tag'])
        team_features = team_features.sort_values('date_str')
        new_features = set()

        def get_slope(s):
            slope, _, _, _, _ = stats.linregress(list(range(s.shape[0])), s.tolist())
            return slope

        for t in tqdm.tqdm(teams):
            for c in self.initial_team_data_columns:
                col_name1 = f'{self.feature_indicator_str}_{self.team_str}_rl_avg_{c}_{n}'
                col_name2 = f'{self.feature_indicator_str}_{self.team_str}_rl_trend_{c}_{n}'

                team_features.loc[team_features['team_tag'] == t, col_name1] = \
                self.team_data[self.team_data['team_tag'] == t].shift(periods=1).rolling(window=n)[c].mean()
                new_features.add(col_name1)

                team_features.loc[team_features['team_tag'] == t, col_name2] = \
                self.team_data[self.team_data['team_tag'] == t].shift(periods=1).rolling(window=n)[c].apply(get_slope)
                new_features.add(col_name2)

        new_features = list(new_features)

        team_features = team_features[new_features + ['team_game_key']]
        team_features = team_features.set_index('team_game_key')
        self.feature_df = self.feature_df.join(team_features)

        self.standalone_feature_columns.extend(new_features)
        self.diff_feature_cols.extend(new_features)

        self.standalone_feature_columns = sorted(list(set(self.standalone_feature_columns)))
        self.diff_feature_cols = sorted(list(set(self.diff_feature_cols)))

    @timeit
    def build_event_features(self):
        print('build_event_features start: {}'.format(self.feature_df.isna().sum().sum()))
        team_data_cols = set(self.team_data.columns)
        feature_cols = set(self.feature_df.columns)

        team_data_cols = sorted(
            list(team_data_cols & set(self.standalone_feature_columns + self.key_columns + self.diff_feature_cols)))
        feature_cols = sorted(
            list(feature_cols & set(self.standalone_feature_columns + self.key_columns + self.diff_feature_cols)))

        print(team_data_cols)
        print(self.key_columns)
        print(set(self.team_data.columns))

        all_teams_data = self.team_data[team_data_cols]
        all_feature_data = self.feature_df[feature_cols]

        rows_dicts_team_data = dict()
        for _, row in all_teams_data.iterrows():
            rows_dicts_team_data[(row['game_key'], row['team_tag'], row['opponent_tag'])] = row[team_data_cols]
        rows_dicts_features = dict()
        for _, row in all_feature_data.iterrows():
            rows_dicts_features[(row['game_key'], row['team_tag'], row['opponent_tag'])] = row[feature_cols]

        print('build_event_features', 2, self.team_data.shape, self.feature_df.shape)
        merge_cols = ['team_game_key']

        results = list()
        for _, row in tqdm.tqdm(all_teams_data.iterrows()):
            next_dict = dict()

            features_record = rows_dicts_features[(row['game_key'], row['team_tag'], row['opponent_tag'])]
            features_opponent_record = rows_dicts_features[(row['game_key'], row['opponent_tag'], row['team_tag'])]
            opponent_row = rows_dicts_team_data[(row['game_key'], row['opponent_tag'], row['team_tag'])]

            if self.fill_nans:
                features_record = features_record.fillna(0)
                features_opponent_record = features_opponent_record.fillna(0)
                opponent_row = opponent_row.fillna(0)
                row = row.fillna(0)

            next_dict['team_game_key'] = row['team_game_key']

            for i in self.standalone_feature_columns:
                if i in team_data_cols and i not in feature_cols:
                    next_dict[i] = row[i]
                    merge_cols.append(i)

            for i in self.diff_feature_cols:
                if i in feature_cols:
                    r1 = features_record[i]
                    r2 = features_opponent_record[i]
                    next_dict['{}_diff'.format(i)] = r1 - r2
                    merge_cols.append('{}_diff'.format(i))
                elif i in team_data_cols:
                    r1 = row[i]
                    r2 = opponent_row[i]
                    next_dict['{}_diff'.format(i)] = r1 - r2
                    merge_cols.append('{}_diff'.format(i))
            results.append(next_dict)
        merge_cols = list(set(merge_cols))

        print('build_event_features', 3, self.team_data.shape, self.feature_df.shape)
        new_feature_df = pd.DataFrame.from_dict(results)
        new_feature_df = new_feature_df[merge_cols + ['team_game_key']]
        new_feature_df = new_feature_df.set_index('team_game_key')
        self.feature_df = self.feature_df.join(new_feature_df)
        print('build_event_features end: {}'.format(self.feature_df.isna().sum().sum()))



    @timeit
    def assign_home_for_teams(self):
        team_data_copy = self.team_data.copy()
        home_dict = find_team_home_loc(team_data_copy)
        team_data_copy['feature_home'] = team_data_copy.apply(
            lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis=1)
        self.standalone_feature_columns.append('feature_home')
        team_data_copy = team_data_copy[['team_game_key', 'feature_home']]
        team_data_copy = team_data_copy.set_index('team_game_key')
        self.feature_df = self.feature_df.join(team_data_copy)

    @timeit
    def assign_date_since_last_game(self):
        team_data_copy = self.team_data.copy()
        team_data_copy = team_data_copy.sort_values('date_str')
        team_data_copy['date_dt'] = pd.to_datetime(team_data_copy['date_str'], errors='coerce')
        team_data_copy['days_since_last_match'] = team_data_copy.groupby('team_tag')['date_dt'].diff()
        team_data_copy['days_since_last_match'] = team_data_copy['days_since_last_match'].dt.days
        self.standalone_feature_columns.append('days_since_last_match')
        self.diff_feature_cols.append('days_since_last_match')
        team_data_copy = team_data_copy.set_index('team_game_key')
        self.feature_df = self.feature_df.join(team_data_copy[['days_since_last_match']])

    @timeit
    def encode_dates(self):
        team_data_copy = self.team_data.copy()
        team_data_copy['date_dt'] = pd.to_datetime(self.team_data['date_str'], errors='coerce')
        team_data_copy['dow'] = team_data_copy['date_dt'].dt.dayofweek
        team_data_copy['year'] = team_data_copy['date_dt'].dt.year
        team_data_copy['month'] = team_data_copy['date_dt'].dt.month

        merge_cols = ['team_game_key']
        for e in ['dow', 'year', 'month']:
            for i in set(team_data_copy[e]):
                team_data_copy['{0}_{1}'.format(e, i)] = team_data_copy[e].apply(lambda x: x == i).astype(int)
                self.standalone_feature_columns.append('{0}_{1}'.format(e, i))
                merge_cols.append('{0}_{1}'.format(e, i))
        merge_cols = list(set(merge_cols))

        team_data_copy = team_data_copy[merge_cols]
        team_data_copy = team_data_copy.set_index('team_game_key')
        self.feature_df = self.feature_df.join(team_data_copy)

    @timeit
    def calculate_team_game_rating(self, rating_type):
        team_data_copy = self.team_data.sort_values('date_str').copy()
        'feature_team_pregame_rating_0'
        new_col_pre = f'feature_team_pregame_rating_{rating_type}'
        new_col_post = f'feature_team_postgame_rating_{rating_type}'
        self.diff_feature_cols.append(new_col_pre)
        self.standalone_feature_columns.append(new_col_pre)

        team_data_copy = team_data_copy[['team_game_key', 'team_tag', 'opponent_tag', 'date_str', 'game_key']]
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


        team_data_copy = team_data_copy.set_index('team_game_key')
        self.feature_df = self.feature_df.join(team_data_copy[[new_col_pre]])


    def process_minutes_played(self):
        self.team_data['mp'] = self.team_data['mp'].apply(parse_minutes_played)
        self.player_data['mp'] = self.player_data['mp'].apply(parse_minutes_played)


    #################################################################################################################
    # Helper methods

    @timeit
    def fillna(self):
        for i in sorted(
                list(set(self.initial_team_data_columns + self.standalone_feature_columns + self.diff_feature_cols))):
            if i == self.target:
                continue
            if i in self.team_data.columns:
                self.team_data[i] = self.team_data[i].fillna(self.team_data[i].median())
                self.team_data[i] = self.team_data[i].fillna(0)
            if i in self.feature_df.columns:
                self.feature_df[i] = self.feature_df[i].fillna(self.feature_df[i].median())
                self.feature_df[i] = self.feature_df[i].fillna(0)

    @timeit
    def scale_data(self):
        self.scaler_dict = dict()
        for i in sorted(
                list(set(self.initial_team_data_columns + self.initial_player_data_columns + self.standalone_feature_columns + self.diff_feature_cols))):
            if i == self.target:
                continue
            if i in self.team_data.columns and i in self.feature_df.columns:
                print('col {} in both raw and features'.format(i))
            if i in self.team_data.columns:
                scaler = StandardScaler()
                self.team_data[i] = scaler.fit_transform(
                    self.team_data[i].fillna(self.team_data[i].median()).values.reshape(-1, 1))
                self.scaler_dict[(i, 'team_data')] = scaler
            if i in self.player_data.columns:
                scaler = StandardScaler()
                self.player_data[i] = scaler.fit_transform(
                    self.player_data[i].fillna(self.player_data[i].median()).values.reshape(-1, 1))
                self.scaler_dict[(i, 'player_data')] = scaler
            if i in self.feature_df.columns:
                print(i)
                scaler = StandardScaler()
                self.feature_df[i] = scaler.fit_transform(
                    self.feature_df[i].fillna(self.feature_df[i].median()).values.reshape(-1, 1))
                self.scaler_dict[(i, 'feature_df')] = scaler


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

    @timeit
    def save_past_n_game_dataset_combined(self, past_n_game_datasets_combined, history_length,
                                team_encoding_size,
                                team_encoding_type,
                                player_encoding_size_1,
                                player_encoding_size_2,
                                player_encoding_type):
        with open(
                f'{data_path}/{past_n_game_dataset_combined_table_name}_{history_length}_{self.fill_nans}_{self.data_scaling}_{team_encoding_size}_{team_encoding_type}_{player_encoding_size_1}_{player_encoding_size_2}_{player_encoding_type}.pkl',
                'wb') as f:
            pickle.dump(past_n_game_datasets_combined, f)

    @timeit
    def load_past_n_game_dataset_combined(self, history_length,
                                team_encoding_size,
                                team_encoding_type,
                                player_encoding_1_size,
                                player_encoding_2_size,
                                player_encoding_type):
        with open(
                f'{data_path}/{past_n_game_dataset_combined_table_name}_{history_length}_{self.fill_nans}_{self.data_scaling}_{team_encoding_size}_{team_encoding_type}_{player_encoding_1_size}_{player_encoding_2_size}_{player_encoding_type}.pkl',
                'rb') as f:
            return pickle.load(f)


    @timeit
    def save_past_n_game_dataset(self, past_n_game_dataset, history_length):
        with open(
                f'{data_path}/{past_n_game_dataset_table_name}_{history_length}_{self.fill_nans}_{self.data_scaling}.pkl',
                'wb') as f:
            pickle.dump(past_n_game_dataset, f)

    @timeit
    def load_past_n_game_dataset(self, history_length):
        with open(
                f'{data_path}/{past_n_game_dataset_table_name}_{history_length}_{self.fill_nans}_{self.data_scaling}.pkl',
                'rb') as f:
            return pickle.load(f)

    @timeit
    def save_processed_data(self):
        self.team_data.to_csv(
            '{data_path}/{db_name}_processed_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path,
                                                                                    db_name=box_score_details_table_name,
                                                                                    data_scaling=self.data_scaling,
                                                                                    fill_nans=self.fill_nans
                                                                                    ),
            sep='|', index=False)
        self.player_data.to_csv(
            '{data_path}/{db_name}_processed_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path,
                                                                                    db_name=player_detail_table_name,
                                                                                    data_scaling=self.data_scaling,
                                                                                    fill_nans=self.fill_nans
                                                                                    ), sep='|',
            index=False)

    @timeit
    def load_processed_data(self):
        self.team_data = pd.read_csv(
            '{data_path}/{db_name}_processed_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path,
                                                                                    db_name=box_score_details_table_name,
                                                                                    data_scaling=self.data_scaling,
                                                                                    fill_nans=self.fill_nans),
            sep='|', low_memory=False)
        self.player_data = pd.read_csv(
            '{data_path}/{db_name}_processed_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path,
                                                                                    db_name=player_detail_table_name,
                                                                                    data_scaling=self.data_scaling,
                                                                                    fill_nans=self.fill_nans),
            sep='|', low_memory=False)

    @timeit
    def save_columns(self):
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='key_columns'), 'w') as f:
            json.dump(self.key_columns, f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='standalone_feature_columns'),
                  'w') as f:
            json.dump(self.standalone_feature_columns, f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='diff_feature_cols'), 'w') as f:
            json.dump(self.diff_feature_cols, f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='initial_team_column_list'),
                  'w') as f:
            json.dump(self.initial_team_data_columns, f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='initial_player_column_list'),
                  'w') as f:
            json.dump(self.initial_player_data_columns, f)

    @timeit
    def load_columns(self):
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='key_columns'), 'r') as f:
            self.key_columns = json.load(f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='standalone_feature_columns'),
                  'r') as f:
            self.standalone_feature_columns = json.load(f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='diff_feature_cols'), 'r') as f:
            self.diff_feature_cols = json.load(f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='initial_team_column_list'),
                  'r') as f:
            self.initial_team_data_columns = json.load(f)
        with open('{data_path}/{db_name}.json'.format(data_path=data_path, db_name='initial_player_column_list'),
                  'r') as f:
            self.initial_player_data_columns = json.load(f)

    @timeit
    def save_feature_df(self):
        self.feature_df.to_csv('{data_path}/{db_name}_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path,
                                                                                             db_name='features',
                                                                                             data_scaling=self.data_scaling,
                                                                                             fill_nans=self.fill_nans),
                               sep='|', index=True)
        # print('save_feature_df: {}'.format(self.feature_df.columns.tolist()))

    @timeit
    def load_feature_df(self):
        self.feature_df = pd.read_csv(
            '{data_path}/{db_name}_{fill_nans}_{data_scaling}.csv'.format(data_path=data_path, db_name='features',
                                                                          data_scaling=self.data_scaling,
                                                                          fill_nans=self.fill_nans), sep='|',
            low_memory=False)
        # print('load_feature_df: {}'.format(self.feature_df.columns.tolist()))


@timeit
def get_dense_autoencoder(input_shape,
                          bottleneck_size,
                          dense_layer_activations,
                          bottleneck_layer_activations,
                          target_activations):
    input_layer = layers.Input(shape=input_shape)
    encoder = layers.Dense(128, activation=dense_layer_activations)(input_layer)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(64, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(32, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(16, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder_out = layers.Dense(bottleneck_size, activation=bottleneck_layer_activations, name='bottleneck')(encoder)
    decoder = layers.Dense(16, activation=dense_layer_activations)(encoder_out)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(32, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(64, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(128, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    out = layers.Dense(input_shape[0], activation=target_activations)(decoder)

    autoencoder = models.Model(input_layer, out)
    encoder = models.Model(input_layer, encoder_out)
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return autoencoder, encoder


class Encoder:
    def __init__(self, encoder_type, encoder_dims, encoder_id, encoder_params):
        self.encoder_type = encoder_type
        self.encoder_dims = encoder_dims
        self.encoder_id = encoder_id
        self.encoder_params = encoder_params
        self.model = None
        self.scaler_dict = dict()
        self.file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}'
        self.scaler_file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}_scaler'

    @timeit
    def fit(self, x):


        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        assert len(x.shape) == 2

        x_df = pd.DataFrame(data = x,
                            columns=list(range(x.shape[1])))


        if self.encoder_type == 'pca':
            for i in x_df.columns:
                scaler = StandardScaler()
                if x_df[i].isna().sum() / x_df.shpae[0] == 1.0:
                    x_df[i].fillna(0)
                x_df[i]= scaler.fit_transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
                self.scaler_dict[i] = scaler
            self.model = PCA(n_components=self.encoder_dims)
            self.model.fit(x)

        if self.encoder_type == 'dense_autoencoder':
            for i in x_df.columns:
                scaler = StandardScaler()
                if x_df[i].isna().sum() / x_df.shape[0] == 1.0:
                    x_df[i].fillna(0)
                x_df[i]= scaler.fit_transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
                # x_df[i] *= 2
                # x_df[i] -= 1
                self.scaler_dict[i] = scaler

            self.autoencoder, self.encoder = get_dense_autoencoder(input_shape = (x_df.shape[1],),
                                              bottleneck_size = self.encoder_dims,
                                              dense_layer_activations = 'softplus',
                                              bottleneck_layer_activations = 'linear',
                                              target_activations= 'linear')
            x_train, x_val = train_test_split(x_df, random_state=1)

            early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=1,
                                         verbose=0, mode='auto')
            self.autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), callbacks=[early_stopping],
                           epochs=200, batch_size=128)
            self.save()

            for layer in self.autoencoder.layers:
                print(layer.name)

            for layer in self.encoder.layers:
                print(layer.name)

            self.model_output = self.encoder.predict(x)
            print(self.model_output.shape)
            return self.model_output

    def transform(self, x):
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        assert len(x.shape) == 2

        x_df = pd.DataFrame(data = x,
                            columns=list(range(x.shape[1])))

        if self.encoder_type == 'pca':
            for i in x_df.columns:
                x_df[i] = self.scaler_dict[i].transform(x_df[i].fillna(0).values.reshape(-1, 1))
            return self.model.transform(x_df)
        if self.encoder_type == 'dense_autoencoder':

            for i in x_df.columns:
                x_df[i] = self.scaler_dict[i].transform(x_df[i].fillna(0).values.reshape(-1, 1))

            return self.encoder.predict(x_df)

    @timeit
    def save(self):
        if self.encoder_type == 'pca':
            with open('{file_path}.pkl'.format(file_path=self.file_path), 'wb') as f:
                pickle.dump(self.model, f)

        if self.encoder_type == 'dense_autoencoder':
            models.save_model(self.encoder, '{file_path}_encoder.pkl'.format(file_path=self.file_path))
            models.save_model(self.autoencoder, '{file_path}_autoencoder.pkl'.format(file_path=self.file_path))
        with open('{file_path}.pkl'.format(file_path=self.scaler_file_path), 'wb') as f:
            pickle.dump(self.scaler_dict, f)

    @timeit
    def load(self):
        if self.encoder_type == 'pca':
            with open('{file_path}.pkl'.format(file_path=self.file_path), 'rb') as f:
                self.model = pickle.load(f)
        if self.encoder_type == 'dense_autoencoder':
            self.encoder = models.load_model('{file_path}_encoder.pkl'.format(file_path=self.file_path))
            self.autoencoder = models.load_model('{file_path}_autoencoder.pkl'.format(file_path=self.file_path))
        with open('{file_path}.pkl'.format(file_path=self.scaler_file_path), 'rb') as f:
            self.scaler_dict = pickle.load(f)



def create_data_files():
    dm = DataManager(fill_nans=False, data_scaling=None, testing=1000)
    dm.update_raw_datasets()
    dm.build_timeseries(4, True)
    x1, x2, y, x2_cols = dm.get_labeled_data(4, True, True)
    print(x1.shape, x2.shape, y.shape)


if __name__ == '__main__':
    create_data_files()
