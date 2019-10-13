import pandas as pd

from sherdog_scraper import (base_output_folder
                             )
from common import (parse_list_of_ints_from_str,
                    clean_text,
                    get_new_rating,
                    starting_rating)
import datetime
import tqdm
import functools
import operator
from sklearn.preprocessing import LabelEncoder
import pickle
import os


nan_cat = 'nan_cat'

#######################################################################################################################
# Data cleaning


def convert_height_to_metric(s):
    m_per_ft = .3048
    m_per_in = .0254

    component_list = parse_list_of_ints_from_str(s)
    if len(component_list) == 2:
        return m_per_ft * component_list[0] + m_per_in * component_list[1]


def convert_weight_to_metric(s):
    kg_per_lb = 0.453592

    component_list = parse_list_of_ints_from_str(s)
    if len(component_list) == 1:
        return kg_per_lb * component_list[0]


def clean_name(s):
    s_split = str(s).split('"')
    if len(s_split) >= 3:
        return clean_text(s_split[0] + s_split[-1])
    return clean_text(s)


def get_event_org(event_s):
    event_s = str(event_s)

    if 'UFC' in event_s:
        return 'UFC'
    if 'Invicta' in event_s:
        return 'Invicta'
    if 'Strikeforce' in event_s:
        return 'Strikeforce'

    event_split = event_s.split('-')[0]
    event_split = event_split.split(' ')
    return ' '.join(event_split[:-1])


def parse_event_date(s):
    month_map = {'Jan': 1,
                 'Feb': 2,
                 'Mar': 3,
                 'Apr': 4,
                 'May': 5,
                 'Jun': 6,
                 'Jul': 7,
                 'Aug': 8,
                 'Sep': 9,
                 'Oct': 10,
                 'Nov': 11,
                 'Dec': 12,
                 }

    split_date = str(s).split('/')
    if len(split_date) == 3:
        try:
            month_num = month_map[split_date[0].strip()]
            return datetime.datetime(int(split_date[2]), month_num, int(split_date[1]))
        except ValueError:
            pass


def extract_fight_end_time(end_round, round_end_time):
    assumed_round_time = 300
    seconds_per_min = 60

    components = parse_list_of_ints_from_str(round_end_time)
    if len(components) == 2:
        round_end_time_s = seconds_per_min * components[0] + components[1]
        finished_rounds = end_round - 1
        time_of_previous_rounds = assumed_round_time * finished_rounds
        return round_end_time_s + time_of_previous_rounds


def extract_round_end_time(round_end_time):
    seconds_per_min = 60

    components = parse_list_of_ints_from_str(round_end_time)
    if len(components) == 2:
        round_end_time_s = seconds_per_min * components[0] + components[1]
        return round_end_time_s


def extract_general_method(s):
    split_method = str(s).split('(')
    if len(split_method) >= 2:
        clean_method = clean_text(split_method[0])
    else:
        clean_method = clean_text(s)

    if 'tko' in clean_method or 'ko' in clean_method:
        return 'ko'
    elif 'submission' in clean_method:
        return 'submission'
    elif 'decision' in clean_method:
        return 'decision'
    else:
        return 'other'


def extract_method(s):
    split_method = str(s).split('(')
    if len(split_method) >= 2:
        clean_method = clean_text(split_method[0])
        return clean_method


def extract_details(s):
    split_method = str(s).split('(')
    if len(split_method) >= 2:
        clean_method = clean_text(' '.join([i for i in split_method[1:]]))
        return clean_method


def process_personal_data(df):
    df['birth_dt'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['height_m'] = df['height'].apply(lambda x: convert_height_to_metric(x))
    df['weight_m'] = df['weight'].apply(lambda x: convert_weight_to_metric(x))
    df['name'] = df['sherdog_name'].apply(lambda x: clean_name(x))
    df = df[['fighter_id', 'nationality', 'birth_dt', 'height_m', 'weight_m', 'name']]
    return df


def process_fight_data(df):
    df['event_org'] = df['event_name'].apply(lambda x: get_event_org(x))
    df['fight_dt'] = df['fight_date'].apply(lambda x: parse_event_date(x))
    df['fight_date_str'] = df['fight_dt'].apply(lambda x: str(x))
    df['round_end_time'] = df['fight_end_time'].apply(lambda x: extract_round_end_time(x))
    df['fight_end_time'] = df.apply(lambda x: extract_fight_end_time(x['fight_end_round'], x['fight_end_time']), axis=1)
    df['general_method'] = df['method'].apply(lambda x: extract_general_method(x))
    df['method_details'] = df['method'].apply(lambda x: extract_details(x))
    df['method'] = df['method'].apply(lambda x: extract_method(x))
    df['fight_type'] = df['fight_type_text']
    df['fight_id'] = df.apply(lambda x: clean_text(str(sorted([x['fighter_id'], x['opponent_id'], x['fight_date_str']]))), axis=1)

    res_mapping = {'win': 1,
                   'loss':0}
    df['result'] = df['result'].apply(lambda x: res_mapping.get(str(x).lower(), .5))

    df = df[['event_org', 'fight_dt', 'round_end_time', 'fight_end_time', 'fight_end_round', 'general_method', 'method_details', 'method',
             'fight_type', 'fighter_id', 'opponent_id', 'result', 'fight_id']]
    return df


def prepare_data(run_id=None):
    output_folder = f'{base_output_folder}/{run_id}'
    personal_df = pd.read_csv(f'{output_folder}/personal_data.csv', sep='|')
    fight_df = pd.read_csv(f'{output_folder}/fight_data.csv', sep='|')

    personal_df = process_personal_data(personal_df)
    fight_df = process_fight_data(fight_df)

    personal_df.to_csv(f'{output_folder}/processed_fighter_data.csv', sep = '|', index = False)
    fight_df.to_csv(f'{output_folder}/processed_fight_data.csv', sep = '|', index = False)


######################################################################################################################
# Rating calculation

def get_most_recent_record_before_date(df, date, fighter_id):
    sub_df = df[(df['fight_dt'] < date) & (df['fighter_id'] == fighter_id)]
    if not sub_df.empty:
        return sub_df.iloc[-1]
    return sub_df


def calculate_rating(df, filtered_df, col_name, rating_type = 0):
    print()
    print(f'Calculating ratings: {col_name} {rating_type}')

    base_col_name = f'{col_name}_rating_{rating_type}'
    pre_fight_fighter_col_name = f'{base_col_name}_fighter_pre_fight'
    post_fight_fighter_col_name = f'{base_col_name}_fighter_post_fight'
    pre_fight_opponent_col_name = f'{base_col_name}_opponent_pre_fight'
    post_fight_opponent_col_name = f'{base_col_name}_opponent_post_fight'
    pre_fight_rating_diff_col_name = f'{base_col_name}_pre_fight_rating_diff'

    filtered_df[pre_fight_fighter_col_name] = None
    filtered_df[post_fight_fighter_col_name] = None
    filtered_df[pre_fight_opponent_col_name] = None
    filtered_df[post_fight_opponent_col_name] = None
    filtered_df = filtered_df.sort_values('fight_dt')

    filtered_dicts = filtered_df.to_dict(orient='records')
    for r in tqdm.tqdm(filtered_dicts):
        outcome = r['result']

        previous_fighter_record = get_most_recent_record_before_date(filtered_df, r['fight_dt'], r['fighter_id'])
        previous_opponent_record = get_most_recent_record_before_date(filtered_df, r['fight_dt'], r['opponent_id'])

        pre_fight_fighter_rating = None
        pre_fight_opponent_rating = None

        if not previous_fighter_record.empty:
            pre_fight_fighter_rating = previous_fighter_record[post_fight_fighter_col_name]
        if not previous_opponent_record.empty:
            pre_fight_opponent_rating = previous_opponent_record[post_fight_fighter_col_name]
        if not pre_fight_fighter_rating:
            pre_fight_fighter_rating = starting_rating
        if not pre_fight_opponent_rating:
            pre_fight_opponent_rating = starting_rating

        post_fight_fighter_rating = get_new_rating(pre_fight_fighter_rating, pre_fight_opponent_rating, outcome, rating_type = rating_type)
        post_fight_opponent_rating = get_new_rating(pre_fight_opponent_rating, pre_fight_fighter_rating, 1 if outcome == 0 else 0, rating_type = rating_type)

        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                    filtered_df['fighter_id'] == r['fighter_id']), pre_fight_fighter_col_name] = pre_fight_fighter_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                    filtered_df['fighter_id'] == r['fighter_id']), pre_fight_opponent_col_name] = pre_fight_opponent_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                    filtered_df['fighter_id'] == r['fighter_id']), post_fight_fighter_col_name] = post_fight_fighter_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                    filtered_df['fighter_id'] == r['fighter_id']), post_fight_opponent_col_name] = post_fight_opponent_rating

    filtered_df = filtered_df[['fight_id', 'fighter_id', 'fight_dt', pre_fight_fighter_col_name, post_fight_fighter_col_name, pre_fight_opponent_col_name, post_fight_opponent_col_name]]
    df = df.merge(filtered_df, how = 'left', on = ['fight_id', 'fighter_id', 'fight_dt'])
    df = df[['fight_id', 'fighter_id', 'fight_dt', pre_fight_fighter_col_name, post_fight_fighter_col_name, pre_fight_opponent_col_name, post_fight_opponent_col_name]]

    fighter_ids = set(df['fighter_id'])
    fighter_id_dict = dict()
    for fighter_id in tqdm.tqdm(fighter_ids):
        fighter_id_dict[fighter_id] = df[df['fighter_id'] == fighter_id]

    out_dfs = []
    for fighter_id in tqdm.tqdm(fighter_ids):
        temp_df = fighter_id_dict[fighter_id].copy()
        temp_df = temp_df.sort_values('fight_dt')

        temp_df[pre_fight_fighter_col_name] = temp_df[pre_fight_fighter_col_name].fillna(method='ffill')
        temp_df[post_fight_fighter_col_name] = temp_df[post_fight_fighter_col_name].fillna(method='ffill')
        temp_df[pre_fight_opponent_col_name] = temp_df[pre_fight_opponent_col_name].fillna(method='ffill')
        temp_df[post_fight_opponent_col_name] = temp_df[post_fight_opponent_col_name].fillna(method='ffill')

        temp_df[pre_fight_fighter_col_name] = temp_df[pre_fight_fighter_col_name].fillna(starting_rating)
        temp_df[post_fight_fighter_col_name] = temp_df[post_fight_fighter_col_name].fillna(starting_rating)
        temp_df[pre_fight_opponent_col_name] = temp_df[pre_fight_opponent_col_name].fillna(starting_rating)
        temp_df[post_fight_opponent_col_name] = temp_df[post_fight_opponent_col_name].fillna(starting_rating)

        out_dfs.append(temp_df)

    df = pd.concat(out_dfs)
    df[pre_fight_rating_diff_col_name] = df[pre_fight_fighter_col_name] - df[pre_fight_opponent_col_name]
    return df


def calculate_all_ratings(run_id, min_perc = .1):
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/processed_fight_data.csv', sep='|')
    df = df[['fight_id', 'fighter_id', 'opponent_id', 'result', 'fight_dt', 'general_method', 'event_org', 'method_details']]

    df = df.sort_values('fight_dt')
    rating_dfs = []

    df_copy = df.copy()
    rating_dfs.append(calculate_rating(df, df_copy, 'all', rating_type = 0))
    rating_dfs.append(calculate_rating(df, df_copy, 'all', rating_type = 1))

    rating_subsets = ['method_details', 'event_org', 'general_method']
    for s in rating_subsets:
        value_counts_series = df[s].value_counts(normalize=True)
        for k, v in zip(value_counts_series.index, value_counts_series):
            if v > min_perc:
                df_sub = df[df[s] == k].copy()
                col_name = f'{s}_{k}'
                col_name = clean_text(col_name).replace(' ', '_')
                rating_dfs.append(calculate_rating(df, df_sub, col_name, rating_type = 0))
                rating_dfs.append(calculate_rating(df, df_sub, col_name, rating_type = 1))

    out_df = functools.reduce(pd.merge, rating_dfs)
    out_df.to_csv(f'{output_folder}/fighter_ratings.csv', sep = '|', index = False)


#######################################################################################################################
# Feature extraction


def get_age(birth_date, current_date):
    birth_date_split = parse_list_of_ints_from_str(birth_date)
    current_date_split = parse_list_of_ints_from_str(current_date)

    if len(birth_date_split) == 3 and  len(current_date_split) == 3:
        birth_dt = datetime.datetime(birth_date_split[0], birth_date_split[1], birth_date_split[2])
        current_dt = datetime.datetime(current_date_split[0], current_date_split[1], current_date_split[2])
        return (current_dt - birth_dt).days


def build_personal_features(df):
    df['is_same_nationality'] = df.apply(lambda x: int(x['fighter_nationality'] == x['opponent_nationality']), axis = 1)
    df['fighter_age'] = df.apply(lambda x: get_age(x['fighter_birth_dt'], x['fight_dt']), axis=1)
    df['opponent_age'] = df.apply(lambda x: get_age(x['opponent_birth_dt'], x['fight_dt']), axis=1)
    df['age_diff'] = df['fighter_age'] - df['opponent_age']
    df['height_diff'] = df['fighter_height_m'] - df['opponent_height_m']

    feature_cols = ['is_same_nationality', 'fighter_age', 'opponent_age', 'age_diff', 'height_diff']
    return df, feature_cols


def build_moving_avg_features(df):
    added_cols = set()

    mov_avg_cols_cat_cols = ['general_method']
    mov_avg_cols = ['fight_end_time', 'result', 'fight_end_round']
    window_sizes = [1, 3,  5, 10]

    fighter_ids = set(df['fighter_id'])

    fighter_id_dict = dict()
    for fighter_id in tqdm.tqdm(fighter_ids):
        fighter_id_dict[fighter_id] = df[df['fighter_id'] == fighter_id].shape[0]

    for fighter_id in tqdm.tqdm(fighter_ids):
        for w in window_sizes:
            if fighter_id_dict[fighter_id] <= w:
                continue
            for c in mov_avg_cols:
                df.loc[df['fighter_id'] == fighter_id, f'fighter_moving_average_{c}_{w}'] = df.loc[df['fighter_id'] == fighter_id].shift(periods=1).rolling(window=w)[c].mean()
                df.loc[df['opponent_id'] == fighter_id, f'opponent_moving_average_{c}_{w}'] = df.loc[df['opponent_id'] == fighter_id].shift(periods=1).rolling(window=w)[c].mean()
                added_cols.update({f'fighter_moving_average_{c}_{w}', f'opponent_moving_average_{c}_{w}'})

            for c in mov_avg_cols_cat_cols:
                values = set(df[c])
                for v in values:
                    df['temp_col'] = df[c].apply(lambda x: 1 if x == v else 0).astype(float)
                    df.loc[df['fighter_id'] == fighter_id, f'fighter_moving_average_cat_{c}_{v}_{w}'] = df.loc[df['fighter_id'] == fighter_id].shift(periods=1).rolling(window=w)['temp_col'].mean()
                    df.loc[df['opponent_id'] == fighter_id, f'opponent_moving_average_cat_{c}_{v}_{w}'] = df.loc[df['opponent_id'] == fighter_id].shift(periods=1).rolling(window=w)['temp_col'].mean()
                    added_cols.update({f'fighter_moving_average_cat_{c}_{v}_{w}', f'opponent_moving_average_cat_{c}_{v}_{w}'})

    if 'temp_col' in df.columns:
        df = df.drop('temp_col', axis = 1)

    return df, list(added_cols)


def label_encode_cat_cols(df, run_id):
    pass


def get_ratings(df, run_id):
    output_folder = f'{base_output_folder}/{run_id}'
    rating_df = pd.read_csv(f'{output_folder}/fighter_ratings.csv', sep = '|')
    rating_cols = [i for i in rating_df.columns if i not in df.columns]
    df = df.merge(rating_df)
    return df, rating_cols


def feature_extraction(run_id):
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/processed_fight_data.csv', sep='|')
    fighter_df = pd.read_csv(f'{output_folder}/processed_fighter_data.csv', sep='|')
    fighter_df.columns = [f'fighter_{i}' if 'fighter' not in i else i for i in fighter_df.columns]
    df = df.merge(fighter_df, on = ['fighter_id'])
    fighter_df = pd.read_csv(f'{output_folder}/processed_fighter_data.csv', sep='|')
    fighter_df.columns = [f'opponent_{i}' if 'fighter' not in i else i for i in fighter_df.columns]
    fighter_df['opponent_id'] = fighter_df['fighter_id']
    fighter_df = fighter_df.drop('fighter_id', axis = 1)
    df = df.merge(fighter_df, on = ['opponent_id'])
    df = df.sort_values('fight_dt')
    print(1)

    feature_cols = []
    df, cols = build_personal_features(df)
    feature_cols.extend(cols)

    df, cols = build_moving_avg_features(df)
    feature_cols.extend(cols)

    df, cols = get_ratings(df, run_id)
    feature_cols.extend(cols)

    df = df.reset_index()
    df['record_id'] = df.index

    df.to_csv(f'{output_folder}/all_features_and_targets.csv', index=False, sep='|')

    feature_cols.append('record_id')
    target_cols = ['result', 'general_method', 'fight_end_round', 'record_id']

    features = df[feature_cols]
    targets = df[target_cols]

    features.to_csv(f'{output_folder}/features.csv', index = False, sep = '|')
    targets.to_csv(f'{output_folder}/target.csv', index = False, sep = '|')



if __name__ == '__main__':
    run_id = '2019-10-12_16-19-56'
    prepare_data(run_id=run_id)
    calculate_all_ratings(run_id=run_id)
    feature_extraction(run_id=run_id)
