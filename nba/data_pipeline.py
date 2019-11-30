import pandas as pd
from nba.common import (
    data_path,
    starting_rating,
    get_new_rating,
    timeit,
parse_minutes_played,
box_score_details_table_name,
player_detail_table_name,
box_score_details_table_name_sample,
player_detail_table_name_sample,
processed_team_data_table_name,
processed_player_data_table_name,
general_feature_data_table_name
)
from collections import Counter
import tqdm
import traceback


team_columns_to_aggregate = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                      'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                      'ft_pct', 'fta', 'fta_per_fga_pct', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                      'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                      'ts_pct', 'usg_pct', 'home', 'win', 'days_since_last_match']


#################################################################################################################
# Data processing/cleaning/structuring
def get_raw_data(sample):
    if sample:
        team_df = pd.read_csv(f'{data_path}/{box_score_details_table_name_sample}.csv', sep = '|', low_memory=False)
        player_df = pd.read_csv(f'{data_path}/{player_detail_table_name_sample}.csv', sep = '|', low_memory=False)
    else:
        team_df = pd.read_csv(f'{data_path}/{box_score_details_table_name}.csv', sep = '|', low_memory=False)
        player_df = pd.read_csv(f'{data_path}/{player_detail_table_name}.csv', sep = '|', low_memory=False)
    return team_df, player_df


def save_processed_data(team_df, player_df):
    team_df.to_csv(f'{data_path}/{processed_team_data_table_name}.csv', sep = '|', index = False)
    player_df.to_csv(f'{data_path}/{processed_player_data_table_name}.csv', sep = '|', index = False)


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


def process_minutes_played(df):
    df['mp'] = df['mp'].apply(parse_minutes_played)
    return df


@timeit
def assign_home_for_teams(df):
    home_dict = find_team_home_loc(df)
    df['home'] = df.apply(
        lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis=1)
    return df


@timeit
def assign_date_since_last_game(df):
    df['year_str'] = df['year'].astype(str)
    df['month_str'] = df['month'].astype(str)
    df['day_str'] = df['day'].astype(str)

    df['year_str'] = df['year_str'].str.zfill(4)
    df['month_str'] = df['month_str'].str.zfill(2)
    df['day_str'] = df['day_str'].str.zfill(2)

    df['date_str'] = df['year_str'] + '-' + df['month_str'] + '-' + df['day_str']
    df = df.sort_values('date_str')
    df['date_dt'] = pd.to_datetime(df['date_str'], errors='coerce')
    df['days_since_last_match'] = df.groupby('team_tag')['date_dt'].diff()
    df['days_since_last_match'] = df['days_since_last_match'].dt.days
    return df


@timeit
def calculate_team_game_rating(df, rating_type):
    df = df.sort_values('date_str')
    new_col_pre = f'team_pregame_rating_{rating_type}'
    new_col_post = f'team_postgame_rating_{rating_type}'

    df[new_col_pre] = None
    df[new_col_post] = None

    for i, r in tqdm.tqdm(df.iterrows()):
        team_previous_record = get_most_recent_team_record_before_date(df, r['team_tag'],
                                                                            r['date_str'])
        opponent_previous_record = get_most_recent_team_record_before_date(df, r['opponent_tag'],
                                                                                r['date_str'])

        if team_previous_record.empty or pd.isna(team_previous_record[new_col_post]):
            team_previous_rating = starting_rating
        else:
            team_previous_rating = team_previous_record[new_col_post]

        if opponent_previous_record.empty or pd.isna(opponent_previous_record[new_col_post]):
            opponent_previous_rating = starting_rating
        else:
            opponent_previous_rating = opponent_previous_record[new_col_post]

        df.loc[i, new_col_pre] = team_previous_rating
        df.loc[i, new_col_post] = get_new_rating(
            team_previous_rating,
            opponent_previous_rating,
            r['win'], multiplier=1, rating_type=rating_type)
    df[new_col_pre] = pd.to_numeric( df[new_col_pre])
    return df


def get_most_recent_team_record_before_date(df, tag, date_str):
    sub_df = df[(df['date_str'] < date_str) & (df['team_tag'] == tag)]
    if not sub_df.empty:
        return sub_df.iloc[-1]
    return sub_df


def process_raw_data(sample = False):
    '''
    Create new fields, home game

    Extract simple features.

    Create feature and target dataframe.
    '''
    team_df, player_df = get_raw_data(sample=sample)
    team_df = process_minutes_played(team_df)
    player_df = process_minutes_played(player_df)
    team_df = assign_home_for_teams(team_df)
    team_df = assign_date_since_last_game(team_df)
    team_df = calculate_team_game_rating(team_df, 0)
    team_df = calculate_team_game_rating(team_df, 1)

    for i in team_df.columns:
        print(i, team_df[i].dtype)

    save_processed_data(team_df, player_df)


#################################################################################################################
# General feature engineering

def get_processed_data():
    team_df = pd.read_csv(f'{data_path}/{processed_team_data_table_name}.csv', sep = '|', low_memory=False)
    player_df = pd.read_csv(f'{data_path}/{processed_player_data_table_name}.csv', sep = '|', low_memory=False)
    return team_df, player_df


def save_general_feature_file(feature_df):
    feature_df.to_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep = '|', index = False)


def build_team_aggregates(team_df, history_length_list):
    teams = set(team_df['team_tag'])
    team_df = team_df.sort_values('date_str')
    new_features = set()

    for t in tqdm.tqdm(teams):
        for n in history_length_list:
            for c in team_columns_to_aggregate:
                col_name1 = f'team_{c}_past_{n}_game_avg'
                team_df.loc[team_df['team_tag'] == t, col_name1] = team_df[team_df['team_tag'] == t].shift(periods=1).rolling(window=n)[c].mean()
                new_features.add(col_name1)

    new_features = list(new_features)
    for t in tqdm.tqdm(teams):
        years_active = set(team_df[team_df['team_tag'] == t]['year'])
        sub_team_df = team_df[(team_df['team_tag'] == t)][new_features]

        for y in years_active:
            sub_team_year_df = team_df[(team_df['team_tag'] == t)&(team_df['year'] == y)][new_features]
            for col_name in new_features:
                if sub_team_year_df[col_name].isna().all():
                    if sub_team_df[col_name].isna().all():
                        fill_value = team_df[col_name].median()
                    else:
                        fill_value = sub_team_df[col_name].median()
                else:
                    fill_value = sub_team_year_df[col_name].median()
                team_df.loc[(team_df['team_tag'] == t)&(team_df['year'] == y), col_name] = team_df.loc[(team_df['team_tag'] == t)&(team_df['year'] == y), col_name].fillna(fill_value)

    return team_df, new_features


def add_opponent_features(df, columns_to_compare):
    new_features = []
    df_copy = df.copy()
    df_copy['temp_column'] = df_copy['team_tag']
    df_copy['team_tag'] = df_copy['opponent_tag']
    df_copy['opponent_tag'] = df_copy['temp_column']

    df_copy['team_game_key'] = df_copy.apply(
        lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)
    df['team_game_key'] = df_copy.apply(
        lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

    opponent_columns = []
    for i in columns_to_compare:
        col_name = '{}_opponent'.format(i)
        df_copy[col_name] = df_copy[i]
        opponent_columns.append(col_name)

    df = df.merge(df_copy[['team_game_key'] + opponent_columns])

    diff_features = []
    for i in columns_to_compare:
        try:
            col_name = '{}_diff'.format(i)
            df[col_name] = df[i] - df['{}_opponent'.format(i)]
            diff_features.append('{}_diff'.format(i))
        except:
            traceback.print_exc()

    new_features.extend(new_features)
    new_features.extend(diff_features)
    return df, new_features


def fill_nans(df):
    return df.fillna(df.median())


def process_general_features(categorical_strategy = 'dictionary_encoding'):
    team_features = ['team_pregame_rating_0', 'team_pregame_rating_1', 'days_since_last_match', 'home']
    target = 'win'
    team_df, _ = get_processed_data()
    team_df, new_features = build_team_aggregates(team_df, [3, 5, 25])
    team_features.extend(new_features)
    team_df, new_features = add_opponent_features(team_df, team_features)
    team_features.extend(new_features)

    feature_df = team_df[team_features + [target]]
    feature_df = fill_nans(feature_df)
    save_general_feature_file(feature_df)


#################################################################################################################
# Time series feature engineering


#################################################################################################################
# Data output

def load_general_feature_file():
    feature_df = pd.read_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep = '|', low_memory=False)
    return feature_df


#################################################################################################################
# Test functions

if __name__ == '__main__':
    process_raw_data(sample = False)
    process_general_features()

