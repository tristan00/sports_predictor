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
combined_feature_file_data_table_name
                    )
from sklearn.decomposition import PCA
# import multiprocessing
from collections import Counter
import tqdm
import re
from scipy import stats


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
    feature_indicator_str = 'feature'
    team_str = 'team'
    player_str = 'player'
    opponent_str = 'opponent'
    pregame_rating_str = 'pregame_rating'
    postgame_rating_str = 'postgame_rating'

    initial_team_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                   'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft', 'ft_pct',
                                   'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                   'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct', 'ts_pct',
                                   'usg_pct', f'{feature_indicator_str}_home']
    initial_player_data_columns = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                   'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft', 'ft_pct',
                                   'fta', 'fta_per_fga_pct', 'mp', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                   'plus_minus', 'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct', 'ts_pct',
                                   'usg_pct']
    target = 'win'
    id_columns = ['team_tag', 'team_link', 'team_name', 'opponent_tag', 'opponent_name', 'opponent_link', 'location',
                  'date_str', 'game_key', 'team_game_key', 'player_link']

    pca_dims = 8

    def __init__(self, team_data, player_data):
        self.team_data = team_data
        self.player_data = player_data

        self.team_dfs_dict = dict()
        self.team_dfs_dict = dict()

        self.team_features = pd.DataFrame()
        self.player_features = pd.DataFrame()

        self.team_data['date_str'] = self.team_data.apply(lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis = 1)
        self.team_data['game_key'] = self.team_data.apply(lambda x:  str(sorted([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])])), axis = 1)
        self.team_data['team_game_key'] = self.team_data.apply(lambda x:  str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis = 1)

        self.player_data['date_str'] = self.player_data.apply(lambda x: str(x['year']).zfill(4) + '-' + str(x['month']).zfill(2) + '-' + str(x['day']).zfill(2), axis = 1)
        self.player_data['game_key'] = self.player_data.apply(lambda x:  str(sorted([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])])), axis = 1)
        self.player_data['team_game_key'] = self.player_data.apply(lambda x:  str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis = 1)

        self.team_data = self.team_data.sort_values('date_str')
        self.player_data = self.player_data.sort_values('date_str')


    def process_features(self):
        self.player_data = self.load_player_data()
        self.team_data = self.load_team_data()
        self.player_data = self.player_data.merge(self.team_data)
        cols_to_drop = {'team_link', 'team_name', 'opponent_name', 'opponent_link', 'location', 'player_link'}
        self.player_data = self.player_data.drop(list(cols_to_drop & set(self.player_data.columns)), axis = 1)


        opponent_data = self.player_data.copy()
        opponent_data['temp_col'] = opponent_data['team_tag']
        opponent_data['temp_col'] = opponent_data['team_tag']
        opponent_data['team_tag'] = opponent_data['opponent_tag']
        opponent_data['opponent_tag'] = opponent_data['temp_col']

        print(opponent_data.shape)
        opponent_data['team_game_key'] = opponent_data.apply(lambda x:  str([x['date_str'], x['team_tag'], x['opponent_tag']]), axis = 1)
        opponent_data = opponent_data.set_index(['team_tag', 'opponent_tag', 'team_game_key', 'date_str', 'game_key'])
        self.player_data = self.player_data.set_index(['team_tag', 'opponent_tag', 'team_game_key', 'date_str', 'game_key'])
        opponent_data.columns = ['{0}_opponent'.format(i) for i in opponent_data.columns]

        columns = self.player_data.columns.tolist()
        self.player_data = self.player_data.join(opponent_data)
        for i in columns:
            if i != self.target:
                self.player_data['{0}_difference_over_opponent'.format(i)] = self.player_data[i] - self.player_data['{0}_opponent'.format(i)]
        self.save_output()

    @timeit
    def save_output(self):
        self.player_data.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=combined_feature_file_data_table_name), sep='|', index = False)

    #################################################################################################################
    # Team features
    @timeit
    def create_team_features(self):
        self.assign_home_for_teams()
        self.train_team_vectorizer()
        self.get_past_n_game_data(10)
        for i in [0]:
            self.calculate_team_game_rating(i)
        self.team_data = self.team_data[[i for i in self.team_data.columns if i not in self.initial_team_data_columns]]
        self.save_team_data()


    @timeit
    def train_team_vectorizer(self):
        self.team_game_vectorizer = PCA(n_components=self.pca_dims)

        for i in self.initial_team_data_columns:
            self.team_data[i] = self.team_data[i].replace(np.inf, np.nan).replace(-np.inf, np.nan)
            if self.team_data[i].isna().sum() / self.team_data[i].shape[0] < 1.0:
                self.team_data[i] = self.team_data[i].fillna(self.team_data[i].median())
            self.team_data[i] = self.team_data[i].fillna(0)

        self.team_game_vectorizer.fit(self.team_data[self.initial_team_data_columns])

    @timeit
    def save_team_data(self):
        self.team_data.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=processed_team_data_table_name), sep='|', index = False)

    @timeit
    def load_team_data(self):
        return pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=processed_team_data_table_name), sep='|')

    @timeit
    def assign_home_for_teams(self):
        home_dict = find_team_home_loc(self.team_data)
        self.team_data[f'{self.feature_indicator_str}_home'] = self.team_data.apply(lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis = 1)

    @timeit
    def get_past_n_game_data(self, n):
        self.team_data = self.team_data.set_index(['team_game_key'])
        self.team_data = self.team_data.sort_values('date_str')
        # next_records = self.team_game_vectorizer.transform(self.team_data[self.initial_team_data_columns])

        next_records_df = pd.DataFrame(data=self.team_data[self.initial_team_data_columns],
                                       # columns=[f'{self.feature_indicator_str }_{self.team_str}_past_team_data_pca_dim_{n}' for n in range(self.pca_dims)],
                                       columns=[f'{i}_past_team_data_{n}' for n, i in enumerate(self.initial_team_data_columns)],
                                       index = self.team_data.index)

        for i in range(1, n + 1):
            temp_next_records = next_records_df.shift(i)
            temp_next_records.columns = ['{0}_{1}_records'.format(j, i) for j in temp_next_records.columns]
            self.team_data = self.team_data.join(temp_next_records)

        self.team_data = self.team_data.reset_index()
        self.save_team_data()


    @timeit
    def calculate_team_moving_averages(self, n):
        teams = set(self.team_data['team_tag'])
        for t in teams:
            for c in self.initial_team_data_columns:
                self.team_data.loc[self.team_data['team_tag'] == t, f'{self.feature_indicator_str }_{self.team_str}_rl_avg_{c}_{n}'] = self.team_data[self.team_data['team_tag'] == t].shift(periods=1).rolling(window=n)[c].mean()
        self.save_team_data()

    @timeit
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
                                                                                                                                   r['win'], multiplier = 1, rating_type = rating_type)
        self.save_team_data()

    #################################################################################################################
    # player features
    @timeit
    def create_player_features(self):
        self.process_minutes_played()
        self.train_player_vectorizer()
        self.get_past_n_player_data(10)
        self.aggregate_player_data()
        self.player_data = self.player_data[[i for i in self.player_data.columns if i not in self.initial_player_data_columns]]
        self.save_player_data()

    @timeit
    def train_player_vectorizer(self):
        self.player_game_vectorizer = PCA(n_components=self.pca_dims)
        for i in self.initial_player_data_columns:
            self.player_data[i] = self.player_data[i].replace(np.inf, np.nan).replace(-np.inf, np.nan)
            if self.player_data[i].isna().sum() / self.player_data[i].shape[0] < 1.0:
                self.player_data[i] = self.player_data[i].fillna(self.player_data[i].median())
            self.player_data[i] = self.player_data[i].fillna(0)
        self.player_game_vectorizer.fit(self.player_data[self.initial_player_data_columns])

    @timeit
    def aggregate_player_data(self):
        self.player_data = self.player_data.merge(self.team_data[[i for i in self.team_data.columns if i not in self.initial_player_data_columns]])
        player_data_num = self.player_data.select_dtypes(include=[np.number])
        player_data_num = player_data_num[[i for i in player_data_num.columns if i not in self.initial_player_data_columns and i != self.target]]
        self.player_data = self.player_data[['team_game_key']].join(player_data_num)
        player_data_agg = self.player_data.groupby(['team_game_key']).agg({i: [np.mean, np.median, np.max, np.min, np.var] for i in self.player_data.columns if i not in {'team_game_key'}})

        print(player_data_agg.columns.tolist())
        columns = self.player_data.columns.tolist()
        for i in columns:
            print(i)
            if i == 'team_game_key':
                continue
            for j in ['mean', 'median', 'amax', 'amin', 'var']:
                player_data_agg['{0}_player_aggregate_{1}'.format(i, j)] = player_data_agg[(i, j)]
                player_data_agg = player_data_agg.drop((i, j), axis = 1)

        self.player_data = player_data_agg.reset_index()
        self.save_player_data()

    @timeit
    def process_minutes_played(self):
        self.player_data['minutes_played'] = self.player_data['mp'].apply(lambda x: str(x).split(':')[0])
        self.player_data['seconds_played'] = self.player_data['mp'].apply(lambda x: str(x).split(':')[-1])
        self.player_data['minutes_played'] = pd.to_numeric(self.player_data['minutes_played'], errors='coerce').fillna(0)
        self.player_data['minutes_played'] = pd.to_numeric(self.player_data['seconds_played'], errors='coerce').fillna(0)
        self.player_data['mp'] = self.player_data.apply(lambda x: x['minutes_played'] + (x['minutes_played']/60) if len(str(x['mp']).split(':')) else None, axis = 1)
        self.player_data = self.player_data.drop(['minutes_played', 'seconds_played'], axis = 1)

    @timeit
    def save_player_data(self):
        self.player_data.to_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=processed_player_data_table_name), sep='|', index = False)
    @timeit
    def load_player_data(self):
        return pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=processed_player_data_table_name), sep='|')
    @timeit
    def get_past_n_player_data(self, n):

        self.player_data = self.player_data.set_index(['team_game_key', 'player_link'])
        self.player_data = self.player_data.sort_values('date_str')
        # next_records = self.player_game_vectorizer.transform(self.player_data[self.initial_player_data_columns])
        next_records = self.player_data[self.initial_player_data_columns]
        next_records_df = pd.DataFrame(data=next_records,
                                       # columns=[f'{self.feature_indicator_str }_{self.player_str}_past_team_data_pca_dim_{n}' for n in range(self.pca_dims)],
                                       columns=[f'{i}_past_player_data_{n}' for n, i  in enumerate(self.initial_player_data_columns)],
                                       index = self.player_data.index)
        players = None
        for i in range(1, n + 1):
            temp_next_records = next_records_df.shift(i)
            temp_next_records.columns = ['{0}_{1}_records'.format(j, i) for j in temp_next_records.columns]
            self.player_data = self.player_data.join(temp_next_records)
        self.player_data = self.player_data.reset_index()
        self.save_player_data()



    @timeit
    def calculate_player_moving_averages(self, n_values):
        players = set(self.player_data['player_link'])

        dfs = []
        for t in tqdm.tqdm(players):
            df = self.player_data[self.player_data['player_link'] == t].copy()
            for c in self.initial_player_data_columns:
                for n in n_values:
                    df[f'{self.feature_indicator_str}_{self.player_str}_rl_avg_{c}_{n}'] = df.shift(periods=1).rolling(window=n)[c].mean()
            dfs.append(df)

        self.player_data = pd.concat(dfs)
        self.save_player_data()

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

    def get_most_recent_player_record_before_date(self, tag, date_str):
        sub_df = self.player_data[(self.player_data['date_str'] < date_str) & (self.player_data['player_link'] == tag)]
        if not sub_df.empty:
            return sub_df.iloc[-1]
        return sub_df

    def get_player_record_right_after_date(self, tag, date_str):
        sub_df = self.player_data[(self.player_data['date_str'] > date_str) & (self.player_data['player_link'] == tag)]
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

    print(player_data.columns.tolist())

    dm = DataManager(team_data, player_data)
    dm.create_team_features()
    dm.create_player_features()
    dm.process_features()


if __name__ == '__main__':
    create_data_files()

