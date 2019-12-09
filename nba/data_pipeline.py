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
    general_feature_data_table_name,
    general_feature_scaled_data_table_name,
    team_time_series_file_loc,
    encoded_file_base_name
)
from nba.encoders import Encoder
from collections import Counter
import tqdm
import traceback
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import pickle
import copy
import time
import gc

#################################################################################################################
# Data processing/cleaning/structuring

@timeit
def get_raw_data(sample):
    team_df = pd.read_csv(f'{data_path}/{box_score_details_table_name}.csv', sep='|', low_memory=False)
    player_df = pd.read_csv(f'{data_path}/{player_detail_table_name}.csv', sep='|', low_memory=False)
    print(f'team df shape: {team_df.shape}, player df shape: {player_df.shape}')
    if sample:
        team_df = team_df[(team_df['year'] >= 2019) & (team_df['month'] >= 3)]
        player_df = player_df[(player_df['year'] >= 2019) & (player_df['month'] >= 3)]
        print(f'team df sample shape: {team_df.shape}, player df sample shape: {player_df.shape}')

    return team_df, player_df


@timeit
def save_processed_data(team_df, player_df):
    team_df.to_csv(f'{data_path}/{processed_team_data_table_name}.csv', sep='|', index=False)
    player_df.to_csv(f'{data_path}/{processed_player_data_table_name}.csv', sep='|', index=False)


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
    df['home'] = pd.to_numeric(df['home'])
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
    df['key'] = df.apply(lambda x: str((x['date_str'], x['team_tag'], x['opponent_tag'])), axis=1)
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
    df[new_col_pre] = pd.to_numeric(df[new_col_pre])
    return df


def get_most_recent_team_record_before_date(df, tag, date_str):
    sub_df = df[(df['date_str'] < date_str) & (df['team_tag'] == tag)]
    if not sub_df.empty:
        return sub_df.iloc[-1]
    return sub_df


@timeit
def process_raw_data(sample=False):
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
    # team_df = calculate_team_game_rating(team_df, 1)
    # team_df = calculate_team_game_rating(team_df, 2)
    # team_df = calculate_team_game_rating(team_df, 3)
    save_processed_data(team_df, player_df)


#################################################################################################################
# General feature engineering

@timeit
def get_processed_data():
    team_df = pd.read_csv(f'{data_path}/{processed_team_data_table_name}.csv', sep='|', low_memory=False)
    player_df = pd.read_csv(f'{data_path}/{processed_player_data_table_name}.csv', sep='|', low_memory=False)
    return team_df, player_df


@timeit
def save_general_feature_file(feature_df, feature_df_scaled):
    feature_df.to_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep='|', index=False)
    print('Unscaled features saved.')
    feature_df_scaled.to_csv(f'{data_path}/{general_feature_scaled_data_table_name}.csv', sep='|', index=False)


@timeit
def build_team_aggregates(team_df, history_length_list, team_columns_to_aggregate):
    teams = set(team_df['team_tag'])
    team_df = team_df.sort_values('date_str')
    new_features = set()

    team_df_copy = team_df.copy()
    team_df_copy = team_df_copy.set_index('key')
    team_aggregate_dict = dict()

    for t in tqdm.tqdm(teams):
        team_aggregate_dict[t] = team_df_copy[team_df_copy['team_tag'] == t]

    for t in tqdm.tqdm(teams):
        for n in history_length_list:
            if n > 1:
                temp_avg_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[
                    team_columns_to_aggregate].mean()
                # temp_median_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[
                #     team_columns_to_aggregate].median()
                # temp_min_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[team_columns_to_aggregate].min()
                temp_max_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[team_columns_to_aggregate].max()
                # temp_var_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[team_columns_to_aggregate].var()
                # temp_skew_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[team_columns_to_aggregate].skew()

                temp_avg_df.columns = [f'team_aggregate_past_{n}_game_avg_{c}' for c in temp_avg_df.columns]
                # temp_median_df.columns = [f'team_aggregate_past_{n}_game_median_{c}' for c in temp_median_df.columns]
                # temp_min_df.columns = [f'team_aggregate_past_{n}_game_min_{c}' for c in temp_min_df.columns]
                # temp_max_df.columns = [f'team_aggregate_past_{n}_game_max_{c}' for c in temp_max_df.columns]
                # temp_var_df.columns = [f'team_aggregate_past_{n}_game_var_{c}' for c in temp_var_df.columns]
                # temp_skew_df.columns = [f'team_aggregate_past_{n}_game_skew_{c}' for c in temp_skew_df.columns]

                # temp_avg_df = temp_avg_df.join(temp_median_df)
                # temp_avg_df = temp_avg_df.join(temp_max_df)
                # temp_avg_df = temp_avg_df.join(temp_min_df)
                # temp_avg_df = temp_avg_df.join(temp_median_df)
                # temp_avg_df = temp_avg_df.join(temp_max_df)
                # temp_avg_df = temp_avg_df.join(temp_var_df)
                # temp_avg_df = temp_avg_df.join(temp_skew_df)
                new_features.update(set(temp_avg_df.columns))

                team_aggregate_dict[t] = team_aggregate_dict[t].join(temp_avg_df)
            else:
                temp_avg_df = team_aggregate_dict[t].shift(periods=1).rolling(window=n)[
                    team_columns_to_aggregate].mean()
                temp_avg_df.columns = [f'team_aggregate_past_{n}_game_avg_{c}' for c in temp_avg_df.columns]
                new_features.update(set(temp_avg_df.columns))
                team_aggregate_dict[t] = team_aggregate_dict[t].join(temp_avg_df)

    team_df = pd.concat(list(team_aggregate_dict.values()))
    team_df = team_df.reset_index()

    team_df = team_df.fillna(0)

    new_features = list(new_features)
    return team_df, new_features


@timeit
def compare_features_to_opponent(df, columns_to_compare):
    columns_to_compare = list(set(columns_to_compare))
    start_time = time.time()

    print(f'Building keys on opponent columns: {time.time() - start_time}')
    new_features = []
    df_copy = df.copy()
    df_copy['temp_column'] = df_copy['team_tag']
    df_copy['team_tag'] = df_copy['opponent_tag']
    df_copy['opponent_tag'] = df_copy['temp_column']

    df_copy['team_game_key'] = df_copy.apply(
        lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)
    df['team_game_key'] = df.apply(
        lambda x: str([str(x['date_str']), str(x['team_tag']), str(x['opponent_tag'])]), axis=1)

    print(f'Remapping opponent columns: {time.time() - start_time}')
    opponent_columns = []
    for i in columns_to_compare:
        col_name = '{}_opponent_feature'.format(i)
        temp_array = df_copy[i].astype(float).values
        # print(f'opponent_columns: {i}, {temp_array.shape}')
        df_copy[col_name] = temp_array
        opponent_columns.append(col_name)

    print(f'Merging in opponent columns: {time.time() - start_time}')

    df = df.merge(df_copy[['team_game_key'] + opponent_columns])

    print(f'Adding difference to opponent columns: {time.time() - start_time}')
    diff_features = []
    for i in columns_to_compare:
        col_name = '{}_diff_vs_opponent_feature'.format(i)
        s1 = df[i].astype(float).values
        s2 = df['{}_opponent_feature'.format(i)].astype(float).values
        df[col_name] = s1 - s2
        diff_features.append(col_name)

    new_features.extend(new_features)
    new_features.extend(diff_features)
    return df, new_features


@timeit
def fill_nans(df):
    # TODO: add subset na filling
    return df.fillna(df.median())


@timeit
def scale_data(feature_df, columns_to_scale):
    feature_df_copy = feature_df.copy()
    for i in columns_to_scale:
        scaler = StandardScaler()
        feature_df_copy[i] = scaler.fit_transform(feature_df[i].values.reshape(-1, 1))
    return feature_df_copy


def get_player_game_aggregates(team_df, player_df, player_columns_to_aggregate):
    player_df = player_df.fillna(0)
    player_df_group_avg = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[
        player_columns_to_aggregate].mean()
    player_df_group_var = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[
        player_columns_to_aggregate].var()
    player_df_group_skew = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[
        player_columns_to_aggregate].skew()
    player_df_group_max = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[
        player_columns_to_aggregate].max()
    player_df_group_min = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[
        player_columns_to_aggregate].min()
    # player_df_group_median = player_df.groupby(['team_link', 'opponent_link', 'year', 'month', 'day'])[player_columns_to_aggregate].median()

    player_df_group_avg.columns = [f'player_stats_aggregated_by_game_{i}_avg' for i in player_columns_to_aggregate]
    player_df_group_var.columns = [f'player_stats_aggregated_by_game_{i}_var' for i in player_columns_to_aggregate]
    player_df_group_skew.columns = [f'player_stats_aggregated_by_game_{i}_skew' for i in player_columns_to_aggregate]
    player_df_group_max.columns = [f'player_stats_aggregated_by_game_{i}_max' for i in player_columns_to_aggregate]
    player_df_group_min.columns = [f'player_stats_aggregated_by_game_{i}_min' for i in player_columns_to_aggregate]
    # player_df_group_median.columns = [f'player_stats_aggregated_by_game_{i}_median' for i in player_columns_to_aggregate]

    player_df_group_avg = player_df_group_avg.join(player_df_group_skew)
    player_df_group_avg = player_df_group_avg.join(player_df_group_max)
    player_df_group_avg = player_df_group_avg.join(player_df_group_min)
    player_df_group_avg = player_df_group_avg.join(player_df_group_var)

    # player_df_group_avg = player_df_group_avg.join(player_df_group_var)
    # player_df_group_avg = player_df_group_avg.join(player_df_group_skew)
    # player_df_group_avg = player_df_group_avg.join(player_df_group_max)
    # player_df_group_avg = player_df_group_avg.join(player_df_group_median)
    # player_df_group_avg = player_df_group_avg.join(player_df_group_min)

    new_cols = player_df_group_avg.columns
    team_df = team_df.merge(player_df_group_avg.reset_index())
    return team_df, new_cols


@timeit
def build_encoded_representations(df, columns, dim_list, encoding_type_list):
    df_copy = df.copy()
    df_copy = df_copy.set_index('key')
    for e_type in encoding_type_list:
        for d in dim_list:
            if d:
                encoder = Encoder(e_type, d, f'{encoded_file_base_name}_{e_type}_{d}', None)
                encoder.fit(df_copy[columns].values)
                preds = encoder.transform(df_copy[columns].values)
                pred_df = pd.DataFrame(data=preds,
                                       index=df_copy.index,
                                       columns=[i for i in range(d)])
                print(f'{data_path}/{encoded_file_base_name}_{e_type}_{d}.csv')
                pred_df.to_csv(f'{data_path}/{encoded_file_base_name}_{e_type}_{d}.csv', sep='|', index=True)


@timeit
def process_general_features(aggregation_windows, encoding_sizes, encoding_types):
    team_features = ['team_pregame_rating_0', 'days_since_last_match', 'home', 'year', 'month']
    targets = ['win', 'score_diff']
    # team_columns_to_aggregate = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
    #                                   'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
    #                                   'ft_pct', 'fta', 'fta_per_fga_pct', 'off_rtg', 'orb', 'orb_pct', 'pf',
    #                                   'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
    #                                   'ts_pct', 'usg_pct', 'home', 'win', 'days_since_last_match', 'score_diff']
    #
    # player_columns_to_aggregate = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct',
    #                                'efg_pct', 'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct',
    #                                'fg_pct', 'fga', 'ft', 'ft_pct', 'fta', 'fta_per_fga_pct',
    #                                'mp', 'off_rtg', 'orb', 'orb_pct', 'pf', 'plus_minus', 'pts',
    #                                'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct', 'ts_pct',
    #                                'usg_pct']

    team_columns_to_aggregate = ['home', 'fg3_pct', 'efg_pct', 'fga', 'fg3a_per_fga_pct', 'stl_pct', 'fg_pct', 'pf',
                                 'drb_pct', 'days_since_last_match', 'fta', 'tov_pct', 'orb_pct', 'fg3a', 'fg3',
                                 'ft_pct', 'win', 'orb', 'blk_pct', 'ft', 'ast', 'def_rtg', 'usg_pct', 'ast_pct',
                                 'off_rtg', 'drb', 'trb_pct', 'ts_pct']
    player_columns_to_aggregate = ['drb', 'ast', 'trb_pct', 'blk_pct', 'drb_pct', 'stl_pct', 'fg3', 'fg3a', 'fg3_pct',
                                   'orb_pct', 'ts_pct', 'fta', 'off_rtg', 'plus_minus', 'tov_pct', 'ast_pct', 'ft_pct',
                                   'tov']

    team_df, player_df = get_processed_data()
    team_df, new_cols = get_player_game_aggregates(team_df, player_df, player_columns_to_aggregate)
    print(len(team_columns_to_aggregate), len(new_cols))
    team_columns_to_aggregate.extend(new_cols)
    team_df, new_features = build_team_aggregates(team_df, aggregation_windows, team_columns_to_aggregate)
    team_features.extend(new_features)
    team_df, new_features = compare_features_to_opponent(team_df, team_features)
    team_features.extend(new_features)

    feature_df = team_df[team_features + targets + ['key']]
    build_encoded_representations(feature_df, team_features, encoding_sizes, encoding_types)
    feature_df = fill_nans(feature_df)
    feature_df_scaled = scale_data(feature_df, team_features)
    save_general_feature_file(feature_df, feature_df_scaled)


#################################################################################################################
# Time series features


@timeit
def generate_team_time_series(df, history_length, time_series_cols, use_standard_scaler):
    df, new_features = compare_features_to_opponent(df, time_series_cols)
    temp_time_series_cols = time_series_cols + new_features

    teams = set(df['team_tag'])
    past_n_game_dataset = dict()
    temp_team_df_dict = dict()
    for t in tqdm.tqdm(teams):
        team_data_copy = df[df['team_tag'] == t].copy()
        temp_team_df_dict[t] = team_data_copy

    for t in tqdm.tqdm(teams):
        temp_team_df_dict[t] = temp_team_df_dict[t].sort_values('date_str')

        game_ids = set(temp_team_df_dict[t]['key'])
        temp_team_df_dict[t] = temp_team_df_dict[t].set_index('key')

        for g in game_ids:
            g_eval = eval(g)
            # print(g_eval)
            temp_date_str, temp_team_tag, temp_opponent_tag = g_eval
            opponent_game_id = str((temp_date_str, temp_opponent_tag, temp_team_tag))

            g_iloc = temp_team_df_dict[t].index.get_loc(g)
            pregame_df= temp_team_df_dict[t].iloc[g_iloc - history_length:g_iloc][temp_time_series_cols].fillna(0)
            pregame_df = pregame_df.reindex(sorted(pregame_df.columns), axis=1)
            pregame_data = pregame_df.values
            while pregame_data.shape[0] < history_length:
                new_array = np.array([[0 for _ in temp_time_series_cols]])
                pregame_data = np.vstack([new_array, pregame_data])

            past_n_game_dataset[g] = {'columns':pregame_df.columns.tolist(),
                                      'data':pregame_data,
                                      'opponent_game_id':opponent_game_id}

    past_n_game_dataset_output = dict()
    for k, v in past_n_game_dataset.items():
        team_data = np.copy(v['data'])
        opponent_data = np.copy(past_n_game_dataset[v['opponent_game_id']]['data'])
        diff_data = team_data - opponent_data
        out_data = np.hstack([team_data, opponent_data, diff_data])

        team_columns = copy.deepcopy(v['columns'])
        opponent_columns = ['{}_time_series_opponent_values'.format(i) for i in copy.deepcopy(v['columns'])]
        diff_columns = ['{}_time_series_opponent_diff'.format(i) for i in copy.deepcopy(v['columns'])]
        out_columns = team_columns + opponent_columns + diff_columns

        past_n_game_dataset_output[k] = {'data':out_data,
                                         'columns':out_columns}

    output = list()
    for k, v in past_n_game_dataset_output.items():
        output.append((k, v))

    with open(f'{data_path}/{team_time_series_file_loc}_{history_length}_{use_standard_scaler}.pkl', 'wb') as f:
        pickle.dump(output, f)


@timeit
def generate_time_series_features(history_lengths):
    '''
    :param history_lengths:
    :param data_choices: raw, all, diff
    :return:
    '''
    team_df, _ = get_processed_data()

    team_columns_to_aggregate = ['ast', 'ast_pct', 'blk', 'blk_pct', 'def_rtg', 'drb', 'drb_pct', 'efg_pct',
                                      'fg', 'fg3', 'fg3_pct', 'fg3a', 'fg3a_per_fga_pct', 'fg_pct', 'fga', 'ft',
                                      'ft_pct', 'fta', 'fta_per_fga_pct', 'off_rtg', 'orb', 'orb_pct', 'pf',
                                      'pts', 'stl', 'stl_pct', 'tov', 'tov_pct', 'trb', 'trb_pct',
                                      'ts_pct', 'usg_pct', 'home', 'win', 'days_since_last_match', 'score_diff']
    # team_columns_to_aggregate = ['efg_pct', 'drb', 'fg3a_per_fga_pct', 'fg3', 'tov', 'fga', 'off_rtg', 'trb_pct',
    #                              'ast_pct', 'ast', 'fta_per_fga_pct', 'stl_pct', 'fg3_pct', 'orb', 'orb_pct', 'fg',
    #                              'ft_pct', 'plus_minus', 'blk_pct', 'ts_pct', 'trb', 'fg3a', 'pf']
    time_series_cols = team_columns_to_aggregate + ['team_pregame_rating_0', 'day', 'month', 'year']
    team_df_scaled = scale_data(team_df, time_series_cols)

    for history_length in history_lengths:
        generate_team_time_series(team_df_scaled.copy(), history_length, time_series_cols, True)
        generate_team_time_series(team_df.copy(), history_length, time_series_cols, False)


#################################################################################################################
# Data output


@timeit
def load_general_feature_file(use_standard_scaler=False):
    if use_standard_scaler:
        feature_df = pd.read_csv(f'{data_path}/{general_feature_scaled_data_table_name}.csv', sep='|', low_memory=False)
    else:
        feature_df = pd.read_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep='|', low_memory=False)
    return feature_df


def load_time_series_data_and_target(history_lengths, target):

    feature_df = pd.read_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep='|', low_memory=False)
    feature_df = feature_df.sort_values('key')

    #keys should be stored by date
    feature_df_train = feature_df.iloc[:int(feature_df.shape[0]*.8)]
    feature_df_val = feature_df.iloc[int(feature_df.shape[0]*.8):int(feature_df.shape[0]*.9)]
    feature_df_test = feature_df.iloc[int(feature_df.shape[0]*.9):]

    output_dict_base = {'y_train': feature_df_train[target],
                   'y_val': feature_df_val[target],
                   'y_test': feature_df_test[target]}

    output_dict = dict()
    for history_length in history_lengths:
        for scaled in [True, False]:
            print(history_length, scaled)
            with open(f'{data_path}/{team_time_series_file_loc}_{history_length}_{scaled}.pkl', 'rb') as f:
                temp_time_series = pickle.load(f)
            temp_time_series = sorted(temp_time_series, key=lambda x: x[0])
            output_dict_copy = copy.deepcopy(output_dict_base)

            output_dict_copy['x_train'] = list()
            output_dict_copy['x_val'] = list()
            output_dict_copy['x_test'] = list()

            for n, i in enumerate(temp_time_series):
                next_data = i[1]['data']
                next_columns = i[1]['columns']
                next_df = pd.DataFrame(data = next_data,
                                       columns = next_columns)
                if n < feature_df_train.shape[0]:
                    output_dict_copy['x_train'].append(next_df)
                elif n < feature_df_train.shape[0] + feature_df_val.shape[0]:
                    output_dict_copy['x_val'].append(next_df)
                else:
                    output_dict_copy['x_test'].append(next_df)
            del temp_time_series
            gc.collect()
            #
            # output_dict_copy['x_train'] = [i[1] for i in temp_time_series[:feature_df_train]]
            # output_dict_copy['x_val'] = [i[1] for i in temp_time_series[int(feature_df.shape[0]*.8):int(feature_df.shape[0]*.9)]]
            # output_dict_copy['x_test'] = [i[1] for i in temp_time_series[int(feature_df.shape[0]*.9):]]

            output_dict[(history_length, scaled)] = output_dict_copy

    return output_dict



@timeit
def load_all_feature_files(history_lengths, timeseries_data_choices, general_features_encoding_lengths):
    output_dict = dict()
    feature_df = pd.read_csv(f'{data_path}/{general_feature_data_table_name}.csv', sep='|', low_memory=False)
    feature_df = feature_df.sort_values('key')

    feature_scaled_df = pd.read_csv(f'{data_path}/{general_feature_scaled_data_table_name}.csv', sep='|',
                                    low_memory=False)
    feature_scaled_df = feature_scaled_df.sort_values('key')
    for d in timeseries_data_choices:
        for h in history_lengths:
            for i in [True, False]:
                with open(f'{data_path}/{team_time_series_file_loc}_{h}_{i}_{d}.pkl', 'rb') as f:
                    temp_time_series = pickle.load(f)
                temp_time_series = sorted(temp_time_series, key=lambda x: x[0])
                output_dict[f'time_series_{h}_{i}_{d}'] = np.array([i[1] for i in temp_time_series])

    for i in general_features_encoding_lengths:
        if i:
            df = pd.read_csv(f'{data_path}/{encoded_file_base_name}_pca_{i}.csv', sep='|', low_memory=False)
            df = df.set_index('key')
            df = df.sort_index()
            output_dict[f'encoded_general_features_{i}'] = df

    output_dict['general_features'] = feature_df
    output_dict['general_features_scaled'] = feature_scaled_df
    return output_dict


#################################################################################################################
# Test functions


def run_pipeline(aggregation_windows,
                 encoding_sizes,
                 encoding_types,
                 history_lengths):
    # process_raw_data(sample = sample)
    # process_general_features(aggregation_windows, encoding_sizes, encoding_types)
    generate_time_series_features(history_lengths)


if __name__ == '__main__':
    run_pipeline(aggregation_windows=[7, 75],
                 encoding_sizes=[16, 128],
                 encoding_types=['pca'],
                 history_lengths=[64])
