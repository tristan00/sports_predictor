import pandas as pd

from sherdog_scraper import (base_output_folder,
                             run_scrape
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
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import uuid
import time

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

    namespace = uuid.uuid4()

    df['fight_id'] = df.apply(
        lambda x: uuid.uuid5(namespace, str(sorted([x['fighter_id'], x['opponent_id'], x['fight_date_str']]))).hex,
        axis=1)
    df['fighter_matchup_id'] = df.apply(lambda x: uuid.uuid5(namespace, str([x['fighter_id'], x['opponent_id']])).hex,
                                        axis=1)
    df['matchup_id'] = df.apply(lambda x: uuid.uuid5(namespace, str(sorted([x['fighter_id'], x['opponent_id']]))).hex,
                                        axis=1)
    df['record_id'] = df.apply(lambda x: uuid.uuid5(namespace, str(
        [x['fighter_id'], x['opponent_id'], x['fight_date_str'], x['fight_counter']])).hex, axis=1)

    res_mapping = {'win': 1.0,
                   'loss': 0.0}
    df['result'] = df['result'].apply(lambda x: res_mapping.get(str(x).lower(), .5))

    df = df[['event_org', 'fight_dt', 'round_end_time', 'fight_end_time', 'fight_end_round', 'general_method',
             'method_details', 'method',
             'fight_type', 'fighter_id', 'opponent_id', 'result', 'fight_id', 'record_id', 'fighter_matchup_id']]
    return df


def prepare_data(run_id=None, sample=False):
    print('running prepare_data')
    output_folder = f'{base_output_folder}/{run_id}'
    if sample:
        personal_df = pd.read_csv(f'{output_folder}/personal_data.csv', sep='|', nrows=1000)
        fight_df = pd.read_csv(f'{output_folder}/fight_data.csv', sep='|', nrows=1000)
    else:
        personal_df = pd.read_csv(f'{output_folder}/personal_data.csv', sep='|')
        fight_df = pd.read_csv(f'{output_folder}/fight_data.csv', sep='|')
    print('prepare_data loaded files: {0} {1}'.format(personal_df.shape, fight_df.shape))

    personal_df = process_personal_data(personal_df)
    fight_df = process_fight_data(fight_df)
    # fighter_ids = set(personal_df['fighter_id'])
    # fight_df = fight_df[(fight_df['fighter_id'].isin(fighter_ids)) & (fight_df['opponent_id'].isin(fighter_ids))]

    personal_df.to_csv(f'{output_folder}/processed_fighter_data.csv', sep='|', index=False)
    fight_df.to_csv(f'{output_folder}/processed_fight_data.csv', sep='|', index=False)
    print(fight_df.columns.tolist())
    print('finished preparing data: {0} {1}'.format(personal_df.shape, fight_df.shape))


######################################################################################################################
# Rating calculation

def get_most_recent_record_before_date(df, date, fighter_id):
    sub_df = df[(df['fight_dt'] < date) & (df['fighter_id'] == fighter_id)]
    if not sub_df.empty:
        return sub_df.iloc[-1]
    return sub_df


def save_rating_data(df, filtered_df, filtered_dicts, iteration, rating_id, output_folder):
    start_time = time.time()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(f'{output_folder}/temp_ratings'):
        os.mkdir(f'{output_folder}/temp_ratings')
    df.to_csv(f'{output_folder}/temp_ratings/{rating_id}_df.csv', sep='|', index=False)
    filtered_df.to_csv(f'{output_folder}/temp_ratings/{rating_id}_filtered_df.csv', sep='|', index=False)
    with open(f'{output_folder}/temp_ratings/{rating_id}_filtered_dicts.pkl', 'wb') as f:
        pickle.dump(filtered_dicts, f)
    with open(f'{output_folder}/temp_ratings/{rating_id}_iteration.pkl', 'wb') as f:
        pickle.dump(iteration, f)
    print(f'saving data at iteration {iteration}, took: {time.time() - start_time} seconds')



def calculate_rating(df, filtered_df, col_name, rating_id, rating_type, use_saved_data, output_folder,
                     save_frequency):
    print()
    print(f'Calculating ratings: {col_name} {rating_type}')

    print(df.columns.tolist())
    base_col_name = f'{col_name}_rating_{rating_type}'
    pre_fight_fighter_col_name = f'{base_col_name}_fighter_pre_fight'
    post_fight_fighter_col_name = f'{base_col_name}_fighter_post_fight'
    pre_fight_opponent_col_name = f'{base_col_name}_opponent_pre_fight'
    post_fight_opponent_col_name = f'{base_col_name}_opponent_post_fight'
    pre_fight_rating_diff_col_name = f'{base_col_name}_pre_fight_rating_diff'

    if use_saved_data and os.path.exists(f'{output_folder}/temp_ratings/{rating_id}_df.csv') and \
            os.path.exists(f'{output_folder}/temp_ratings/{rating_id}_filtered_df.csv') and \
            os.path.exists(f'{output_folder}/temp_ratings/{rating_id}_filtered_dicts.pkl') and \
            os.path.exists(f'{output_folder}/temp_ratings/{rating_id}_iteration.pkl'):
        df = pd.read_csv(f'{output_folder}/temp_ratings/{rating_id}_df.csv', sep='|')
        filtered_df = pd.read_csv(f'{output_folder}/temp_ratings/{rating_id}_filtered_df.csv', sep='|')
        with open(f'{output_folder}/temp_ratings/{rating_id}_filtered_dicts.pkl', 'rb') as f:
            filtered_dicts = pickle.load(f)
        with open(f'{output_folder}/temp_ratings/{rating_id}_iteration.pkl', 'rb') as f:
            iteration = pickle.load(f)

        print(f'resuming rating calculation at iteration {iteration}')

    else:
        filtered_df[pre_fight_fighter_col_name] = None
        filtered_df[post_fight_fighter_col_name] = None
        filtered_df[pre_fight_opponent_col_name] = None
        filtered_df[post_fight_opponent_col_name] = None
        filtered_df = filtered_df.sort_values('fight_dt')
        filtered_dicts = filtered_df.to_dict(orient='records')
        iteration = 0

    for r in tqdm.tqdm(filtered_dicts[iteration:]):
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

        post_fight_fighter_rating = get_new_rating(pre_fight_fighter_rating, pre_fight_opponent_rating, outcome,
                                                   rating_type=rating_type)
        post_fight_opponent_rating = get_new_rating(pre_fight_opponent_rating, pre_fight_fighter_rating,
                                                    1.0 if outcome == 0.0 else 0.0, rating_type=rating_type)

        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                filtered_df['fighter_id'] == r['fighter_id']), pre_fight_fighter_col_name] = pre_fight_fighter_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                filtered_df['fighter_id'] == r['fighter_id']), pre_fight_opponent_col_name] = pre_fight_opponent_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                filtered_df['fighter_id'] == r['fighter_id']), post_fight_fighter_col_name] = post_fight_fighter_rating
        filtered_df.loc[(filtered_df['fight_id'] == r['fight_id']) & (
                filtered_df['fighter_id'] == r[
            'fighter_id']), post_fight_opponent_col_name] = post_fight_opponent_rating

        iteration += 1
        if iteration % save_frequency == 0:
            save_rating_data(df, filtered_df, filtered_dicts, iteration, rating_id, output_folder)

    save_rating_data(df, filtered_df, filtered_dicts, iteration, rating_id, output_folder)

    filtered_df = filtered_df[
        ['fight_id', 'fighter_id', 'fight_dt', pre_fight_fighter_col_name, post_fight_fighter_col_name,
         pre_fight_opponent_col_name, post_fight_opponent_col_name]]
    df = df.merge(filtered_df, how='left', on=['fight_id', 'fighter_id', 'fight_dt'])
    df = df[['record_id', 'fight_id', 'fighter_id', 'fight_dt', pre_fight_fighter_col_name, post_fight_fighter_col_name,
             pre_fight_opponent_col_name, post_fight_opponent_col_name]]

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
    df = df[['record_id', pre_fight_fighter_col_name, pre_fight_opponent_col_name, pre_fight_rating_diff_col_name]]
    return df


def calculate_all_ratings(run_id, min_perc=.04, methods=(0,), use_saved_data=False, save_frequency = 10000):
    print('calculate_all_ratings')
    output_folder = f'{base_output_folder}/{run_id}'

    rating_dfs = []
    df = pd.read_csv(f'{output_folder}/processed_fight_data.csv', sep='|')
    df = df[['record_id', 'fight_id', 'fighter_id', 'opponent_id', 'result', 'fight_dt', 'general_method', 'event_org',
             'method_details']]

    df = df.sort_values('fight_dt')

    for m in methods:
        df_copy = df.copy()
        rating_id = uuid.uuid5(uuid.NAMESPACE_DNS, f'all_{m}')
        rating_dfs.append(calculate_rating(df, df_copy, 'all', rating_id, rating_type=m, use_saved_data=use_saved_data,
                                           output_folder=output_folder, save_frequency=save_frequency))

        rating_subsets = ['method_details', 'event_org', 'general_method']
        for s in rating_subsets:
            value_counts_series = df[s].value_counts(normalize=True)

            print(f'Subset: {s} \n Normalized: {dict( df[s].value_counts(normalize=True))}, \n Counts: {dict( df[s].value_counts(normalize=False))} \n')
            print(dict( df[s].value_counts(normalize=False)))

            for k, v in zip(value_counts_series.index, value_counts_series):
                if v > min_perc:
                    print(s, k, v)
                    df_sub = df[df[s] == k].copy()
                    col_name = f'{clean_text(s)}_{clean_text(k)}'.replace(' ', '_')
                    rating_id = uuid.uuid5(uuid.NAMESPACE_DNS, f'{col_name}_{m}')
                    rating_dfs.append(
                        calculate_rating(df, df_sub, col_name, rating_id, rating_type=m, use_saved_data=use_saved_data,
                                         output_folder=output_folder, save_frequency=save_frequency))

    out_df = rating_dfs[0]
    for c, i in enumerate(rating_dfs[1:]):
        out_df = out_df.merge(i)

    print(out_df.shape)
    out_df.to_csv(f'{output_folder}/fighter_ratings.csv', sep='|', index=False)


#######################################################################################################################
# Feature extraction


def get_age(birth_date, current_date):
    birth_date_split = parse_list_of_ints_from_str(birth_date)
    current_date_split = parse_list_of_ints_from_str(current_date)

    if len(birth_date_split) == 3 and len(current_date_split) == 3:
        birth_dt = datetime.datetime(birth_date_split[0], birth_date_split[1], birth_date_split[2])
        current_dt = datetime.datetime(current_date_split[0], current_date_split[1], current_date_split[2])
        return (current_dt - birth_dt).days


def merge_fighter_data(run_id):
    print('merge_fighter_data')
    output_folder = f'{base_output_folder}/{run_id}'

    df = pd.read_csv(f'{output_folder}/processed_fight_data.csv', sep='|')
    print(df.shape)
    fighter_df = pd.read_csv(f'{output_folder}/processed_fighter_data.csv', sep='|')
    fighter_df.columns = [f'fighter_{i}' if 'fighter' not in i else i for i in fighter_df.columns]
    df = df.merge(fighter_df, on=['fighter_id'])

    fighter_df = pd.read_csv(f'{output_folder}/processed_fighter_data.csv', sep='|')
    fighter_df.columns = [f'opponent_{i}' if 'fighter' not in i else i for i in fighter_df.columns]
    fighter_df['opponent_id'] = fighter_df['fighter_id']
    fighter_df = fighter_df.drop('fighter_id', axis=1)
    df = df.merge(fighter_df, on=['opponent_id'])
    df.to_csv(f'{output_folder}/merged_fight_data.csv', sep='|', index=False)
    print(df.shape)


def build_personal_features(run_id):
    print('build_personal_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')

    le = LabelEncoder()
    df['fighter_nationality'] = le.fit_transform(df['nationality'])
    df['fighter_age'] = df.apply(lambda x: get_age(x['fighter_birth_dt'], x['fight_dt']), axis=1)
    df['fighter_height'] = df.apply(lambda x: get_age(x['fighter_birth_dt'], x['fight_dt']), axis=1)

    df = df[['record_id', 'is_same_nationality', 'fighter_age', 'opponent_age', 'age_diff', 'height_diff']]
    df.to_csv(f'{output_folder}/personal_features.csv', sep='|', index=False)


def build_date_features(run_id):
    print('build_date_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')
    df['fight_dt2'] = pd.to_datetime(df['fight_dt'], errors='coerce')
    df['fight_day_of_week'] = df['fight_dt2'].dt.dayofweek
    df['fight_year'] = df['fight_dt2'].dt.year
    df['fight_month'] = df['fight_dt2'].dt.month
    df = df[['record_id', 'fight_day_of_week', 'fight_year', 'fight_month']]
    df.to_csv(f'{output_folder}/date_features.csv', sep='|', index=False)


def build_fight_timing_features(run_id):
    print('build_fight_timing_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')
    df['fight_dt2'] = pd.to_datetime(df['fight_dt'], errors='coerce')
    df = df.sort_values('fight_dt2')
    df['fighter_days_since_last_fight'] = df.groupby('fighter_id')['fight_dt2'].diff().dt.days
    df = df[['record_id', 'days_since_last_fight']]
    df.to_csv(f'{output_folder}/fight_timing_features.csv', sep='|', index=False)


def build_rematch_features(run_id):
    print('build_rematch_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')
    df = df.sort_values('fight_dt')
    df['fight_dt2'] = pd.to_datetime(df['fight_dt'], errors='coerce')

    all_matchups = dict(df['fighter_matchup_id'].value_counts())
    for m, c in all_matchups.items():
        if c == 1:
            continue

        df.loc[df['fighter_matchup_id'] == m, 'fighter_days_since_last_fight_in_matchup'] = df.groupby(['matchup_id', 'fighter_id'])['fight_dt2'].diff().dt.days
        df.loc[df['fighter_matchup_id'] == m, 'fighter_result_in_last_matchup'] = df.groupby(['matchup_id', 'fighter_id'])['result'].diff()

    df = df[['record_id', 'fighter_days_since_last_fight_in_matchup', 'fighter_result_in_last_matchup']]
    df = df.fillna(df.median())
    df.to_csv(f'{output_folder}/rematch_features.csv', sep='|', index=False)


def merge_initial_features(run_id):
    print('merge_initial_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')
    personal_features = pd.read_csv(f'{output_folder}/personal_features.csv', sep='|')
    date_features = pd.read_csv(f'{output_folder}/date_features.csv', sep='|')
    fight_timing_features = pd.read_csv(f'{output_folder}/fight_timing_features.csv', sep='|')
    rematch_features = pd.read_csv(f'{output_folder}/rematch_features.csv', sep='|')
    rating_features = pd.read_csv(f'{output_folder}/fighter_ratings.csv', sep='|')

    df_out = personal_features.merge(date_features)
    df_out = df_out.merge(fight_timing_features)
    df_out = df_out.merge(rematch_features)
    df_out = df_out.merge(rating_features)

    df_out.to_csv(f'{output_folder}/merged_initial_features.csv', sep='|', index=False)

    df_out = df_out.merge(df)
    df_out.to_csv(f'{output_folder}/merged_initial_features_and_data.csv', sep='|', index=False)


def build_moving_avg_features(run_id, min_perc=.04):
    print('build_moving_avg_features')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_initial_features_and_data.csv', sep='|')
    added_cols = set()

    mov_avg_cols_cat_cols = ['general_method', 'event_org', 'method_details']
    mov_avg_cols = ['fight_end_time', 'result', 'fight_end_round', 'days_since_last_fight']

    for c in mov_avg_cols_cat_cols:
        value_counts_series = df[c].value_counts(normalize=True)
        for v, v_perc in zip(value_counts_series.index, value_counts_series):
            if v_perc >= min_perc:
                col_name = f'cat_{c}_{v}'
                df[col_name] = df[c].apply(lambda x: 1 if x == v else 0)
                mov_avg_cols.append(col_name)

    window_sizes = [1, 2, 3, 5, 8]
    fighter_ids = set(df['fighter_id'])

    fighter_id_dict = dict(df['fighter_id'].value_counts())
    # for fighter_id in tqdm.tqdm(fighter_ids):
    #     fighter_id_dict[fighter_id] = df[df['fighter_id'] == fighter_id].shape[0]

    for fighter_id in tqdm.tqdm(fighter_ids):
        for w in window_sizes:
            if fighter_id_dict[fighter_id] <= w:
                continue
            for c in mov_avg_cols:
                df.loc[df['fighter_id'] == fighter_id, f'fighter_moving_average_{c}_{w}'] = \
                df.loc[df['fighter_id'] == fighter_id].shift(periods=1).rolling(window=w)[c].mean()
                # df.loc[df['opponent_id'] == fighter_id, f'opponent_moving_average_{c}_{w}'] = \
                # df.loc[df['opponent_id'] == fighter_id].shift(periods=1).rolling(window=w)[c].mean()

                df[f'fighter_moving_average_{c}_{w}'] = df[f'fighter_moving_average_{c}_{w}'].fillna(0)
                # df[f'opponent_moving_average_{c}_{w}'] = df[f'opponent_moving_average_{c}_{w}'].fillna(0)

                # df[f'diff_moving_average_{c}_{w}'] = df[f'fighter_moving_average_{c}_{w}'] - df[
                #     f'opponent_moving_average_{c}_{w}']
                # added_cols.update({f'fighter_moving_average_{c}_{w}', f'opponent_moving_average_{c}_{w}',
                #                    f'diff_moving_average_{c}_{w}'})
                added_cols.add(f'fighter_moving_average_{c}_{w}')

    if 'temp_col' in df.columns:
        df = df.drop('temp_col', axis=1)
    df = df[['record_id'] + list(added_cols)]
    df.to_csv(f'{output_folder}/moving_avg_features.csv', sep='|', index=False)


def build_past_opponent_features(run_id):
    print('feature_extraction')
    output_folder = f'{base_output_folder}/{run_id}'
    fight_df = pd.read_csv(f'{output_folder}/merged_fight_data.csv', sep='|')
    fight_df = fight_df[['fighter_id', 'opponent_id', 'record_id']]
    features_fighter = pd.read_csv(f'{output_folder}/merged_initial_features.csv', sep='|')
    features_fighter = features_fighter[[i for i in features_fighter.colums if 'fighter_' in i or '_id' in i]]
    features_opponent = pd.read_csv(f'{output_folder}/merged_initial_features.csv', sep='|')
    features_opponent.columns = [i if 'fighter_' not in i else i.replace('fighter_', 'opponent_') for i in features_opponent.columns]
    features_opponent = features_opponent[[i for i in features_fighter.colums if 'opponent_' in i or '_id' in i]]

    print(fight_df.shape)
    fight_df = fight_df.merge(features_fighter, how = 'left', on = 'record_id')
    print(fight_df.shape)
    fight_df = fight_df.merge(features_fighter, how = 'left', on = 'record_id')
    print(fight_df.shape)

    added_columns = ['record_id']
    for i, j in zip(features_fighter.columns, features_opponent.columns):
        if '_id' not in i and 'fighter' in i:
            new_col_name = f'{i}_sub_by_{j}'
            added_columns.extend([i, j, new_col_name])
            fight_df[new_col_name] = fight_df[i] - fight_df[j]
    fight_df = fight_df[added_columns]
    fight_df.to_csv(f'{output_folder}/combined_fighter_and_opponent_features.csv', sep='|', index=False)


def feature_extraction(run_id):
    print('feature_extraction')
    output_folder = f'{base_output_folder}/{run_id}'
    df = pd.read_csv(f'{output_folder}/merged_initial_features_and_data.csv', sep='|')
    features = pd.read_csv(f'{output_folder}/combined_fighter_and_opponent_features.csv', sep='|')
    moving_avg_features = pd.read_csv(f'{output_folder}/moving_avg_features.csv', sep='|')
    features = features.merge(moving_avg_features)
    targets = df[['result', 'general_method', 'fight_end_round', 'record_id']]
    features.to_csv(f'{output_folder}/final_features.csv', index=False, sep='|')
    targets.to_csv(f'{output_folder}/final_target.csv', index=False, sep='|')
    print(features.shape, targets.shape)


def feature_evaluation(run_id):
    print('feature_evaluation')
    output_folder = f'{base_output_folder}/{run_id}'
    x_df = pd.read_csv(f'{output_folder}/final_features.csv', sep='|')
    y_df = pd.read_csv(f'{output_folder}/final_target.csv', sep='|')

    x_df = x_df.sort_values('record_id')
    y_df = y_df.sort_values('record_id')

    x_df = x_df.drop('record_id', axis=1)
    y = y_df['result']

    x_df = x_df.fillna(x_df.median())

    feature_evaluation = dict()
    for c in x_df.columns:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_df[c], y)
        feature_evaluation[c] = {'column': c,
                                 'slope': slope,
                                 'intercept': intercept,
                                 'r_value': r_value,
                                 'p_value': p_value,
                                 'std_err': std_err
                                 }

    rf = RandomForestRegressor()
    rf.fit(x_df, y)
    for c, v in zip(x_df.columns, rf.feature_importances_):
        feature_evaluation[c]['tree_model_gain'] = v

    feature_evaluation_df = pd.DataFrame.from_dict(list(feature_evaluation.values()))
    feature_evaluation_df.to_csv(f'{output_folder}/feature_evaluation.csv', index=False, sep='|')


def run_data_pipeline(run_id = None, rescrape = False, scrape_iterations = 100):
    if not run_id or rescrape:
        run_id = run_scrape(run_id = run_id, max_iterations=scrape_iterations)
    prepare_data(run_id=run_id, sample=False)
    merge_fighter_data(run_id=run_id)

    use_saved_ratings_data = run_id and not rescrape
    calculate_all_ratings(run_id=run_id, min_perc=.01, use_saved_data=use_saved_ratings_data)

    build_personal_features(run_id=run_id)
    build_date_features(run_id=run_id)
    build_fight_timing_features(run_id=run_id)
    build_rematch_features(run_id=run_id)
    merge_initial_features(run_id=run_id)
    build_moving_avg_features(run_id=run_id, min_perc=.01)
    feature_extraction(run_id=run_id)
    feature_evaluation(run_id=run_id)


def scratch():
    # run_id = None
    # run_id = run_scrape(run_id='2019-10-19_12-35-26', max_iterations=2)

    run_id = '2019-10-19_12-35-26'
    run_id = run_scrape(run_id=run_id, max_iterations=2)

    prepare_data(run_id=run_id, sample=False)
    merge_fighter_data(run_id=run_id)
    calculate_all_ratings(run_id=run_id, min_perc=.01, use_saved_data=True)
    build_personal_features(run_id=run_id)
    build_date_features(run_id=run_id)
    build_fight_timing_features(run_id=run_id)
    build_rematch_features(run_id=run_id)
    merge_initial_features(run_id=run_id)
    build_moving_avg_features(run_id=run_id, min_perc=.01)
    feature_extraction(run_id=run_id)
    feature_evaluation(run_id=run_id)


if __name__ == '__main__':
    scratch()

    # run_id = '2019-10-19_12-35-26'
    # run_data_pipeline(run_id = run_id, rescrape = True, scrape_iterations = 4)
