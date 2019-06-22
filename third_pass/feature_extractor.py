import pandas as pd
import sqlite3
import traceback
from elo import starting_elo, calculate_different_elos
import time
import operator
import pickle
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import multiprocessing
import functools
import glob
import random
import configparser

elo_k_list = [-1, 0, 10, 25, 50, 100, 250, 500, 1000]

starting_elo_dict = {k: starting_elo for k in elo_k_list}
finish_method_list = []
finish_method_details_list = []
max_rounds = 5

config = configparser.ConfigParser()
config.read('properties.ini')
db_location = config.get('mma', 'db_location')
model_location = config.get('mma', 'model_location')


def calculate_new_elo(outcome, player_1_elo, player_2_elo, k=100):
    expected_outcome = player_1_elo/(player_1_elo + player_2_elo)
    new_elo = player_1_elo + (k * (outcome-expected_outcome))
    return new_elo


def get_top_n_methods_and_details(match_df, n):
    try:
        with open(model_location + 'method_list.plk', 'rb') as out_file:
            method_list = pickle.load(out_file)
        with open(model_location + 'method_details_list.plk', 'rb') as out_file:
            method_details_list = pickle.load(out_file)
        return method_list, method_details_list

    except FileNotFoundError:
        methods_dict = dict()
        methods_details_dict = dict()
        for _, i in match_df.iterrows():
            method, method_details = extract_details_from_method(i['result_method'])
            methods_dict.setdefault(method, 0)
            methods_dict[method] += 1
            methods_details_dict.setdefault(method_details, 0)
            methods_details_dict[method_details] += 1

        print()

        method_counts = sorted(methods_dict.items(), key=operator.itemgetter(1), reverse=True)[0:n]
        method_result = {i[0] for i in method_counts}
        method_details_counts = sorted(methods_details_dict.items(), key=operator.itemgetter(1), reverse=True)[0:n]
        method_details_result = {i[0] for i in method_details_counts}

        method_result = sorted(list(method_result))
        method_details_result = sorted(list(method_details_result))

        print(method_result)
        print(method_details_result)

        with open(model_location + 'method_list.plk', 'wb') as out_file:
            pickle.dump(method_result, out_file)
        with open(model_location + 'method_details_list.plk', 'wb') as out_file:
            pickle.dump(method_details_result, out_file)
        return method_details_result, method_result


def extract_details_from_method(method_str):
    if 'decision' in method_str.lower():
        method = 'decision'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'technical submission' in method_str.lower():
        method = 'technical submission'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'submission' in method_str.lower():
        method = 'submission'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'tko' in method_str.lower():
        method = 'tko'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'ko' in method_str.lower():
        method = 'ko'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'nc' in method_str.lower():
        method = 'nc'
        if '(' in method_str and ')' in method_str.split('(')[1]:
            method_detail = method_str.split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    else:
        method = ''
        method_detail = ''
    return method, method_detail


def sql_date_to_date(date_info):
    return datetime.datetime.strptime(date_info, '%Y-%m-%d 00:00:00').date()

def date_to_sql_date(date_info):
    return datetime.datetime.strftime(date_info, '%Y-%m-%d 00:00:00').date()


def extract_features(result_df, match_df, event_date):
    num_of_matches = len(match_df)
    win_rate = len(match_df[match_df['result'] == 'win'])/max(1,num_of_matches)
    method_list = [0 for _ in finish_method_list] + [0]
    method_details_list = [0 for _ in finish_method_details_list] + [0]

    date_list = list(match_df['event_date']) + [event_date]
    date_list = sorted(date_list)
    average_days_between_matches = relativedelta(sql_date_to_date(date_list[-1]), sql_date_to_date(date_list[0])).days/max(1,num_of_matches)

    try:
        elo = look_up_elo_dict(result_df, match_df['fighter_id'].values[0], event_date)
    except:
        elo = starting_elo

    for _, i in match_df.iterrows():
        method, method_details = extract_details_from_method(i['result_method'])
        if method in finish_method_list:
            method_list[finish_method_list.index(method)] += 1
        else:
            method_list[-1] += 1
        if method_details in finish_method_list:
            method_details_list[finish_method_details_list.index(method_details)] += 1
        else:
            method_details_list[-1] += 1

    method_list = [i/max(1,num_of_matches) for i in method_list]
    method_details_list = [i / max(1,num_of_matches) for i in method_details_list]

    round_list = [0 for _ in range(max_rounds)]

    for _, i in match_df.iterrows():
        if i['result_round'] < max_rounds:
            round_list[i['result_round']] += 1
        else:
            round_list[-1] += 1

    round_list = [i / max(1,num_of_matches) for i in round_list]

    feature_list = [elo, average_days_between_matches, num_of_matches, win_rate] + method_list + method_details_list + round_list
    return feature_list


def get_fighter_features(result_df, f1_id, event_date):
    fighter_bool = result_df['fighter_id'] == f1_id
    event_date_bool = result_df['event_date'] < event_date
    f1_df = result_df.loc[fighter_bool & event_date_bool]
    win_df = f1_df[f1_df['result'] == 'win']
    loss_df = f1_df[f1_df['result'] == 'win']

    win_features= extract_features(result_df, win_df, event_date)
    loss_features= extract_features(result_df, loss_df, event_date)
    total_features = extract_features(result_df, f1_df, event_date)
    past_1_event_features = extract_features(result_df, f1_df.head(1), event_date)
    past_2_event_features = extract_features(result_df, f1_df.head(2), event_date)
    past_3_event_features = extract_features(result_df, f1_df.head(3), event_date)
    past_5_event_features = extract_features(result_df, f1_df.head(5), event_date)
    past_10_event_features = extract_features(result_df, f1_df.head(10), event_date)

    return win_features+ loss_features+ total_features+ \
            past_1_event_features + past_2_event_features+\
            past_3_event_features+ past_5_event_features+\
            past_10_event_features


#TODO: replace with loc syntax
def get_result_at_date(result_df, f1_id, event_date):
    fighter_bool = result_df['fighter_id'] == f1_id
    event_date_bool = result_df['event_date'] == event_date
    result = result_df.loc[fighter_bool & event_date_bool]['result'].values[0]
    if result == 'win':
        return [1, 0]
    if result == 'loss':
        return [0, 1]
    else:
        return [0, 0]


def get_features(result_df, f1_id, f2_id, event_date):
    f1_features = get_fighter_features(result_df, f1_id, event_date)
    f2_features = get_fighter_features(result_df, f2_id, event_date)
    f_array = np.array(f1_features + f2_features)
    f_result_array = np.array(get_result_at_date(result_df, f1_id, event_date))
    return {'full_features':f_array, 'result_features':f_result_array}


def get_all_matches(result_df):
    unique_fights = result_df[['fighter_id', 'opponent_id', 'event_date']]
    unique_fights = unique_fights.drop_duplicates()
    return unique_fights


def look_up_elo_dict(result_df, fighter_id, date_played, elo_dict):
    temp_elo = dict()
    try:
        return elo_dict[(fighter_id, date_played)]
    except KeyError:
        last_match = result_df[(result_df['fighter_id'] == fighter_id) & (result_df['event_date']< date_played)].sort_values('event_date', ascending=False).head(1)
        opponent = result_df[(result_df['fighter_id'] == fighter_id) & (result_df['event_date'] == date_played)]['opponent_id'].values

        if len(opponent) > 0:
            opponent_id = opponent[0]
            opponent_matches = result_df[(result_df['fighter_id'] == opponent_id) & (result_df['event_date']< date_played)].sort_values('event_date', ascending=False)
            opponent_last_match = opponent_matches.head(1)
            if len(opponent_last_match) > 0:
                opponent_pre_fight_elo = look_up_elo_dict(result_df, opponent_id, opponent_last_match['event_date'].values[0], elo_dict)
            else:
                opponent_pre_fight_elo = {'pre':starting_elo_dict.copy(), 'post':starting_elo_dict.copy()}

            if len(last_match['event_date'].values) > 0:
                past_fight_elo = look_up_elo_dict(result_df, fighter_id,
                                                     last_match['event_date'].values[0], elo_dict)
                temp_elo['pre'] = past_fight_elo['post']
            else:
                temp_elo['pre'] = starting_elo_dict.copy()

            current_match = result_df[
                (result_df['fighter_id'] == fighter_id) & (result_df['event_date'] == date_played)].head(1)
            if len(current_match) == 0:
                temp_elo['post'] = None
            else:
                if current_match['result'].values[0] == 'win':
                    temp_elo['post'] = calculate_different_elos(1, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                elif current_match['result'].values[0] == 'loss':
                    temp_elo['post'] = calculate_different_elos(0, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                elif current_match['result'].values[0] == 'draw':
                    temp_elo['post'] = calculate_different_elos(.5, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                else:
                    temp_elo['post'] = temp_elo['pre']
            current_result = current_match['result'].values[0]
        else:
            temp_elo = look_up_elo_dict(result_df, fighter_id, last_match['event_date'], elo_dict)
            current_result = None

        elo_dict[(fighter_id, date_played)] = temp_elo
    #print('temp', fighter_id, date_played, temp_elo, current_result)
    return temp_elo


def update_elo_dict(match_df, elo_dict, match_dict, input_list = None):
    match_df = match_df.sample(frac=1)
    start_time = time.time()

    if input_list:
        for i in input_list:
            try:
                print(i['fighter_id'], i['opponent_id'], i['event_date'])
                look_up_elo_dict(match_df, i['fighter_id'], i['event_date'], elo_dict)
                look_up_elo_dict(match_df, i['opponent_id'], i['event_date'], elo_dict)
            except:
                traceback.print_exc()
    else:
        match_df = match_df.sort_values('event_date')
        for count, (i, j) in enumerate(match_df.iterrows()):
            try:
                temp_values = look_up_elo_dict(match_df, j['fighter_id'], j['event_date'], elo_dict)
                print((time.time()-start_time)/(count+1), (time.time()-start_time)/len(elo_dict.keys()), count,
                      len(elo_dict.keys()), j['fighter_id'], j['event_date'], temp_values)

                if count%1000 == 0 and count > 0:
                    pass
                    store_elo_dict(elo_dict)
            except:
                traceback.print_exc()
    return elo_dict


def multiprocess_update_elo_dict(match_df, elo_dict):
    start_time = time.time()
    match_df = match_df.sort_values('event_date')
    unique_dates = match_df['event_date'].unique()
    unique_dates = sorted(unique_dates)
    num_of_cores = multiprocessing.cpu_count()

    manager = multiprocessing.Manager()
    manager_elo_dict = manager.dict()

    processes = []
    for _, i in match_df.iterrows():
        processes.append(
            multiprocessing.Process(target=look_up_elo_dict, args=(match_df, i['fighter_id'], i['event_date'], manager_elo_dict,)))
        processes[-1].start()

        print(i, time.time() - start_time)
    for i in processes:
        i.join()
        print(len(dict(manager_elo_dict).values()))

    #
    # for fight_date in unique_dates:
    #     fighters = match_df.loc[match_df['event_date'] == fight_date]['fighter_id'].unique()
    #     processes = []
    #     for i in fighters:
    #         processes.append(multiprocessing.Process(target=look_up_elo_dict, args=(match_df, i, fight_date, manager_elo_dict, )))
    #     for i in processes:
    #         i.start()
    #     for i in processes:
    #         i.join()
    #     res_dict = dict(manager_elo_dict)
    #     print(fight_date, len(res_dict.values()), (time.time() - start_time)/len(res_dict.values()))
    #     elo_dict = res_dict
    #     store_elo_dict(elo_dict)




def get_data(max_date = None):
    with sqlite3.connect(db_location) as conn:
        if max_date:
            match_df = pd.read_sql('''select * from matches where event_date < ?''', conn, params = (max_date,))
            fighter_df = pd.read_sql('''select * from fighter''', conn)
        else:
            match_df = pd.read_sql('''select * from matches''',conn)
            fighter_df = pd.read_sql('''select * from fighter''', conn)

        match_dict = dict()
        for _, i in match_df.iterrows():
            match_dict[(i['fighter_id'], i['event_date'])] = i

    return match_df, fighter_df, match_dict


def store_elo_dict(elo_dict):
    with open(model_location + 'elo_dict.pkl', 'wb') as infile:
        pickle.dump(elo_dict, infile)


def load_elo_dict():
    try:
        with open(model_location + 'elo_dict.pkl', 'rb') as infile:
            return pickle.load(infile)
    except FileNotFoundError:
        return dict()


def extract_features_df(result_df, match_df, event_date, elo_dict, f_id, name = ''):
    num_of_matches = len(match_df)
    win_rate = len(match_df[match_df['result'] == 'win'])/max(1,num_of_matches)
    method_list = [0 for _ in finish_method_list] + [0]
    method_details_list = [0 for _ in finish_method_details_list] + [0]

    date_list = list(match_df['event_date'])
    date_list = sorted(date_list)
    try:
        average_days_between_matches = relativedelta(sql_date_to_date(date_list[-1]), sql_date_to_date(date_list[0])).days/max(1, num_of_matches)
    except:
        average_days_between_matches = 0

    try:
        event_datetime = sql_date_to_date(event_date)
        elo_list = [elo_dict.get((f_id, i), starting_elo) for i in date_list]
        elos = [elo_list[-1]['post'][i] for i in elo_k_list]
        starting_elos = [elo_list[0]['pre'][i] for i in elo_k_list]

        last_fight = elo_list[-1]
        #last_elos = [elo_list[-1][i] for i in elo_k_list]
        #average_elo_change = [(last_elos[i] - first_elos[i])/len(elo_list) for i in range(len(elo_list))]
        average_elo_change = [(i-j)/len(date_list) for i, j in zip(elos, starting_elos)]
    except:
        #traceback.print_exc()
        elos = [starting_elo for i in elo_k_list]
        starting_elos = [starting_elo for i in elo_k_list]
        average_elo_change = [0 for i in elo_k_list]

    for _, i in match_df.iterrows():
        method, method_details = extract_details_from_method(i['result_method'])
        if method in finish_method_list:
            method_list[finish_method_list.index(method)] += 1
        else:
            method_list[-1] += 1
        if method_details in finish_method_list:
            method_details_list[finish_method_details_list.index(method_details)] += 1
        else:
            method_details_list[-1] += 1

    method_list = [i/max(1,num_of_matches) for i in method_list]
    method_details_list = [i / max(1,num_of_matches) for i in method_details_list]

    round_list = [0 for _ in range(max_rounds)]

    for _, i in match_df.iterrows():
        if i['result_round'] < max_rounds:
            round_list[i['result_round']] += 1
        else:
            round_list[-1] += 1

    round_list = [i / max(1,num_of_matches) for i in round_list]

    df_columns = [name+'average_days_between_matches', name + 'num_of_matches',
                               name + 'win_rate']
    df_columns.extend([name + 'method_' + str(count) for count, _ in enumerate(method_list)])
    df_columns.extend([name + 'method_details_' + str(count) for count, _ in enumerate(method_details_list)])
    df_columns.extend([name + 'round_list_' + str(count) for count, _ in enumerate(round_list)])
    df_columns.extend([name + 'elo_' + str(count) for count, _ in enumerate(elo_k_list)])
    df_columns.extend([name + 'average_elo_change_' + str(count) for count, _ in enumerate(round_list)])
    data = [ average_days_between_matches, num_of_matches, win_rate] + method_list + \
           method_details_list + round_list + elos + average_elo_change
    data_dict = {i:j for i, j in zip(df_columns, data)}
    df = pd.DataFrame.from_dict([data_dict])
    return df


def get_fighter_features_df(result_df, f1_id, event_date, elo_dict, fighter_num_name = ''):
    f1_df = result_df.loc[result_df['fighter_id'] == f1_id]
    pre_math_df = f1_df.loc[f1_df['event_date'] < event_date]
    win_df = pre_math_df.loc[pre_math_df['result'] == 'win']
    loss_df = pre_math_df.loc[pre_math_df['result'] == 'loss']

    win_features= extract_features_df(result_df, win_df, event_date, elo_dict, f1_id, name = fighter_num_name+'_win_')
    loss_features= extract_features_df(result_df, loss_df, event_date, elo_dict, f1_id, name = fighter_num_name+'_loss_')
    total_features = extract_features_df(result_df, pre_math_df, event_date, elo_dict, f1_id, name = fighter_num_name+'_total_')
    past_1_event_features = extract_features_df(result_df, pre_math_df.head(1), event_date, elo_dict, f1_id, name = fighter_num_name+'_past_1_fight_')
    past_2_event_features = extract_features_df(result_df, pre_math_df.head(2), event_date, elo_dict, f1_id, name = fighter_num_name+'_past_2_fight_')
    past_3_event_features = extract_features_df(result_df, pre_math_df.head(3), event_date, elo_dict, f1_id, name = fighter_num_name+'_past_3_fight_')
    past_5_event_features = extract_features_df(result_df, pre_math_df.head(5), event_date, elo_dict, f1_id, name = fighter_num_name+'_past_5_fight_')
    past_10_event_features = extract_features_df(result_df, pre_math_df.head(10), event_date, elo_dict, f1_id, name = fighter_num_name+'_past_10_fight_')

    match_num = pre_math_df.shape[0]
    meta_df = pd.DataFrame.from_dict([{fighter_num_name+'num_of_matches':match_num}])

    df = pd.concat([win_features, loss_features, total_features, past_1_event_features, past_2_event_features, \
                    past_3_event_features, past_5_event_features, past_10_event_features], axis=1)

    return df, meta_df


def get_result_at_date_df(result_df, f1_id, event_date):
    fighter_bool = result_df['fighter_id'] == f1_id
    event_date_bool = result_df['event_date'] == event_date
    result = result_df.loc[fighter_bool & event_date_bool]['result'].values[0]
    if result == 'win':
        return pd.DataFrame.from_dict([{'f1_result': 1, 'f2_result':0}])
    if result == 'loss':
        return pd.DataFrame.from_dict([{'f1_result': 0, 'f2_result':1}])
    else:
        return pd.DataFrame.from_dict([{'f1_result': 0, 'f2_result':0}])


def get_fight_features_df(result_df, f1_id, f2_id, event_date, elo_dict, training= True):
    f1_features, meta_1 = get_fighter_features_df(result_df, f1_id, event_date, elo_dict, fighter_num_name='f1')
    f2_features, meta_2 = get_fighter_features_df(result_df, f2_id, event_date, elo_dict, fighter_num_name='f2')

    meta_df = pd.concat([meta_1, meta_2, pd.DataFrame.from_dict([{'fight_year': sql_date_to_date(event_date).year}])], axis = 1)

    if training:
        f_result_array = get_result_at_date_df(result_df, f1_id, event_date)
    else:
        f_result_array = pd.DataFrame.from_dict([{'f1_result': np.nan, 'f2_result':np.nan}])
    df = pd.concat([f1_features, f2_features, meta_df, f_result_array], axis = 1)
    #df = f1_features.append([f2_features, f_result_array])
    return df


def get_stored_dfs():
    files = glob.glob(model_location +'training_sets/*.plk')
    dfs = []
    for count, i in enumerate(files):
        dfs.append(pd.read_pickle(i))
        break

    return pd.concat(dfs)


def get_fightlist_features_df(input_list, fight_date):
    global finish_method_list
    global finish_method_details_list
    #match_df = get_stored_dfs()
    match_df, fighter_df, match_dict = get_data(fight_date)
    finish_method_list, finish_method_details_list = get_top_n_methods_and_details(match_df, 25)

    elo_dict = load_elo_dict()
    #update_elo_dict(match_df, elo_dict, input_list)
    #store_elo_dict(elo_dict)

    #unique_fights = get_all_matches(match_df)
    feature_list = []
    #unique_fights = unique_fights.sample(frac=1)
    start = time.time()

    result_features = []

    for i in input_list:
        temp_features = get_fight_features_df(match_df, i['f1'], i['f2'], fight_date, elo_dict, training=False)

        temp_result = i.copy()
        temp_result.update({'result':temp_features})
        result_features.append(temp_result)
        print(time.time() - start)
    return result_features


def get_feature_df(max_date = None):
    global finish_method_list
    global finish_method_details_list

    match_df, fighter_df, match_dict = get_data(max_date)
    elo_dict = load_elo_dict()
    #elo_dict = update_elo_dict(match_df, elo_dict)
    elo_dict = update_elo_dict(match_df, elo_dict, match_dict)
    store_elo_dict(elo_dict)
    finish_method_list, finish_method_details_list = get_top_n_methods_and_details(match_df, 25)
    unique_fights = get_all_matches(match_df).sort_values('event_date')
    feature_list = []
    #unique_fights = unique_fights.sample(frac=1)
    start = time.time()
    count = unique_fights.shape[0]
    for count, (_, i) in enumerate(unique_fights.iterrows()):

        try:
            temp_features = get_fight_features_df(match_df, i['fighter_id'], i['opponent_id'], i['event_date'], elo_dict)
            print(count, i['fighter_id'], i['opponent_id'], i['event_date'], (time.time() - start)/max(1, count))
            feature_list.append(temp_features)
        except:
            traceback.print_exc()
            print('error with {0}, {1}, {2}'.format(i['fighter_id'], i['opponent_id'], i['event_date']))
            print()
        if count % 10000 == 0 and count > 0:
            feature_df = pd.concat(feature_list)
            feature_df.to_pickle(model_location + 'training_sets/stored_features_{0}_df.plk'.format(count))
            feature_list = []
            elo_dict = load_elo_dict()
        if count > len(elo_dict.keys()):
            break

    feature_df = pd.concat(feature_list)
    feature_df.to_pickle(model_location + 'training_sets/stored_features_{0}_df.plk'.format(count))


if __name__ == '__main__':
    # match_df, fighter_df, match_dict = get_data(None)
    # elo_dict = load_elo_dict()
    # elo_dict = update_elo_dict(match_df, elo_dict, match_dict)
    # store_elo_dict(elo_dict)
    get_feature_df()
