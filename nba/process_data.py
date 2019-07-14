import pandas as pd
import numpy as np
from common import (
                    data_path,
                    box_score_link_table_name,
                    box_score_details_table_name,
                    player_detail_table_name,
                    starting_rating,
                    pad_num,
                    get_new_rating,
                    file_lock
                    )
# import multiprocessing
from collections import Counter


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

def assign_home(df):
    home_dict = find_team_home_loc(df)
    df['stat_is_home'] = df.apply(lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis = 1)
    return df


def get_most_recent_record_before_date(df, tag, date_str, validate_record_cols = None):
    sub_df = df[(df['date_str'] < date_str) & (df['team_tag'] == tag)]
    if  validate_record_cols:
        sub_df = sub_df.dropna(subset = validate_record_cols)
    if not sub_df.empty:
        # sub_df = sub_df.sort_values('date_str')

        return sub_df.iloc[-1]
    return sub_df


def get_next_record_before_date(df, tag, date_str, validate_record_cols = None):
    sub_df = df[(df['date_str'] > date_str) & (df['team_tag'] == tag)]
    if  validate_record_cols:
        sub_df = sub_df.dropna(subset = validate_record_cols)
    if not sub_df.empty:
        # sub_df = sub_df.sort_values('date_str')
        return sub_df.iloc[-1]
    return sub_df


class DataManager():
    def __init__(self, team_data, player_data):
        self.team_data = team_data
        print(self.team_data.columns.tolist())
        self.player_data = player_data
        self.team_features = pd.DataFrame()
        self.team_data['date_str'] = self.team_data.apply(lambda x: pad_num(x['year'], 4) + '-' + pad_num(x['month'], 2) + '-' + pad_num(x['day'], 2), axis = 1)
        self.team_data['game_key'] = self.team_data.apply(lambda x:  str(sorted([x['date_str'], x['team_tag'], x['opponent_tag']])), axis = 1)

        self.team_data = self.team_data.sort_values(['date_str'])

        self.calculate_moving_averages(1)
        self.calculate_moving_averages(10)

        self.calculate_game_ratings_on_subset(self.team_data, 'total_0_', rating_type = 0)
        self.calculate_game_ratings_on_subset(self.team_data, 'total_1_', rating_type = 1)
        self.calculate_game_ratings_on_subset(self.team_data, 'total_2_', rating_type = 2)
        self.calculate_game_ratings_on_subset(self.team_data, 'total_3_', rating_type = 3)

        # self.calculate_game_ratings_on_subset(self.team_data[self.team_data['stat_is_home'] == 1], 'home')
        # self.calculate_game_ratings_on_subset(self.team_data[self.team_data['stat_is_home'] == 0], 'away')

        self.calculate_team_features()

        self.team_data.to_csv('{data_path}/team_rating_data.csv'.format(data_path=data_path), sep='|',index=False)
        self.team_features.to_csv('{data_path}/team_features.csv'.format(data_path=data_path), sep='|',index=False)


    def calculate_moving_averages(self, n):
        teams = set(self.team_data['team_tag'])
        cols1 = ['stat_ast', 'stat_ast_pct','stat_blk', 'stat_blk_pct','stat_def_rtg','stat_drb','stat_drb_pct','stat_efg_pct','stat_fg',
                  'stat_fg3','stat_fg3_pct','stat_fg3a','stat_fg3a_per_fga_pct','stat_fg_pct','stat_fga','stat_ft','stat_ft_pct',
                  'stat_fta','stat_fta_per_fga_pct','stat_mp','stat_off_rtg','stat_orb','stat_orb_pct','stat_pf','stat_plus_minus',
                  'stat_pts','stat_stl','stat_stl_pct','stat_tov', 'stat_tov_pct', 'stat_trb','stat_trb_pct', 'stat_ts_pct',
                  'stat_usg_pct','win']

        for t in teams:
            for c in cols1:
                self.team_data.loc[self.team_data['team_tag'] == t, 'rl_avg_{col_name}_{window_size}'.format(window_size=n, col_name = c)] = self.team_data[self.team_data['team_tag'] == t].shift(periods=1).rolling(window=n)[c].mean()

    def calculate_player_ratings_on_subset(self, df_subset, prefix):
        self.player_data['{prefix}_pregame_rating'.format(prefix=prefix)] = np.nan
        self.player_data['{prefix}_postgame_rating'.format(prefix=prefix)] = np.nan
        df_subset['{prefix}_pregame_rating'.format(prefix=prefix)] = np.nan
        df_subset['{prefix}_postgame_rating'.format(prefix=prefix)] = np.nan


    def calculate_game_ratings_on_subset(self, df_subset, prefix, rating_type = 0):
        self.team_data['{prefix}_pregame_rating'.format(prefix=prefix)] = np.nan
        self.team_data['{prefix}_postgame_rating'.format(prefix=prefix)] = np.nan
        df_subset['{prefix}_pregame_rating'.format(prefix=prefix)] = np.nan
        df_subset['{prefix}_postgame_rating'.format(prefix=prefix)] = np.nan

        for i, j in df_subset.iterrows():
            tag_1 = j['team_tag']
            tag_2 = j['opponent_tag']
            date_str = j['date_str']

            print(date_str, tag_1, tag_2)
            last_record_1 = get_most_recent_record_before_date(df_subset, tag_1, date_str, validate_record_cols = ['{prefix}_postgame_rating'.format(prefix=prefix)])
            last_record_2 = get_most_recent_record_before_date(df_subset, tag_2, date_str, validate_record_cols = ['{prefix}_postgame_rating'.format(prefix=prefix)])
            next_record_1 = get_most_recent_record_before_date(df_subset, tag_1, date_str)

            if last_record_1.empty:
                old_rating_1 = starting_rating
            else:
                old_rating_1 = last_record_1['{prefix}_postgame_rating'.format(prefix=prefix)]
            if last_record_2.empty:
                old_rating_2 = starting_rating
            else:
                old_rating_2 = last_record_2['{prefix}_postgame_rating'.format(prefix=prefix)]

            new_rating_1 = get_new_rating(old_rating_1, old_rating_2, j['win'], rating_type = rating_type)
            self.team_data.loc[(self.team_data['date_str'] == j['date_str']) & (self.team_data['team_tag'] == tag_1), '{prefix}_pregame_rating'.format( prefix=prefix)] = old_rating_1
            df_subset.loc[(df_subset['date_str'] == j['date_str']) & (df_subset['team_tag'] == tag_1), '{prefix}_pregame_rating'.format( prefix=prefix)] = old_rating_1
            self.team_data.loc[(self.team_data['date_str'] >= j['date_str']) & (self.team_data['team_tag'] == tag_1), '{prefix}_postgame_rating'.format(prefix=prefix)] = new_rating_1
            df_subset.loc[(df_subset['date_str'] >= j['date_str']) & (df_subset['team_tag'] == tag_1), '{prefix}_postgame_rating'.format(prefix=prefix)] = new_rating_1

            if not next_record_1.empty:
                self.team_data.loc[(self.team_data['date_str'] <= next_record_1['date_str']) & (self.team_data['date_str'] > j['date_str']) & (self.team_data['team_tag'] == tag_1), '{prefix}_pregame_rating'.format( prefix=prefix)] = new_rating_1
                self.team_data.loc[(self.team_data['date_str'] <= next_record_1['date_str']) & (self.team_data['date_str'] > j['date_str']) & (self.team_data['team_tag'] == tag_1), '{prefix}_pregame_rating'.format( prefix=prefix)] = new_rating_1

    def calculate_team_features(self):
        output = []

        for data_index, r in self.team_data.iterrows():
            records = self.team_data[self.team_data['game_key'] == r['game_key']]
            r1_1 = records.iloc[0].copy().to_frame().transpose()
            r1_2 = records.iloc[1].copy().to_frame().transpose()
            r2_1 = records.iloc[1].copy().to_frame().transpose()
            r2_2 = records.iloc[0].copy().to_frame().transpose()
            r1_1.columns = ['team_feature_' + i if (('stat' in i and 'stat' != i[0:4]) or '_pregame_rating' in i) else i for i in r1_1.columns]
            r1_2.columns = ['opponent_feature_' + i if ('stat' in i and 'stat' != i[0:4]) or '_pregame_rating' in i else i for i in r1_2.columns]
            r1_2 = r1_2[[i for i in r1_2.columns if 'opponent_feature_' in i]]
            r1_1.index = [data_index]
            r1_2.index = [data_index]
            r1 = r1_1.join(r1_2)

            r2_1.columns = ['team_feature_' + i if (('stat' in i and 'stat' != i[0:4]) or '_pregame_rating' in i) else i
                            for i in r2_1.columns]
            r2_2.columns = [
                'opponent_feature_' + i if ('stat' in i and 'stat' != i[0:4]) or '_pregame_rating' in i else i for i in
                r2_2.columns]
            r2_2 = r2_2[[i for i in r2_2.columns if 'opponent_feature_' in i]]
            r2_1.index = [data_index]
            r2_2.index = [data_index]
            r2 = r2_1.join(r2_2)
            output.extend([r1, r2])

        self.team_features = pd.concat(output)



def create_data_files():
    with file_lock:
        team_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                   db_name=box_score_details_table_name),
                                sep='|', low_memory=False)
        # team_data = team_data[team_data['year'] > 2018]
        player_data = pd.read_csv('{data_path}/{db_name}.csv'.format(data_path=data_path,
                                                                     db_name=player_detail_table_name),
                                  sep='|', low_memory=False)

    team_data = assign_home(team_data)
    DataManager(team_data, player_data)

if __name__ == '__main__':
    create_data_files()

