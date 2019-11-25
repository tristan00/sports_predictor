import pandas as pd
import sqlite3
import traceback
from elo import starting_elo, calculate_different_elos
import time
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import operator
import datetime

max_iter = 1000000
lgbm_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    "learning_rate": 0.01,
    "max_depth": -1,
    'num_leaves':31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    'bagging_freq': 1,
    }

fill_value = -1.0
pound_conversion = 2.20462
path = r'C:\Users\trist\Documents\db_loc\mma/mma3/'
db_location = r'C:\Users\trist\Documents\db_loc\mma/mma_tap.db'
elo_k_list = [10, 25, 50, 100, 250, 500, 1000]

starting_elo_dict = {k: starting_elo for k in elo_k_list}


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
                if current_match['result'].values[0] == 'Win':
                    temp_elo['post'] = calculate_different_elos(1, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                elif current_match['result'].values[0] == 'Loss':
                    temp_elo['post'] = calculate_different_elos(0, temp_elo['pre'],
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


def calculate_elo_df(match_df, input_list = None, name = ''):
    elo_dict = dict()
    match_df = match_df.sample(frac=1)
    start_time = time.time()

    match_df = match_df.set_index(['fighter_id', 'event_date'], drop = False)

    if input_list:
        for i in input_list:
            try:
                # print(i['fighter_id'], i['opponent_id'], i['event_date'])
                look_up_elo_dict(match_df, i['fighter_id'], i['event_date'], elo_dict)
                look_up_elo_dict(match_df, i['opponent_id'], i['event_date'], elo_dict)
            except:
                traceback.print_exc()
    else:
        match_df = match_df.sort_values('event_date')
        print('calculating elos')
        for  i, j in tqdm.tqdm(match_df.iterrows(), total=match_df.shape[0]):
            try:
                temp_values = look_up_elo_dict(match_df, j['fighter_id'], j['event_date'], elo_dict)
                # print((time.time()-start_time)/(count+1), (time.time()-start_time)/len(elo_dict.keys()), count,
                #       len(elo_dict.keys()), j['fighter_id'], j['event_date'], temp_values)

                # if count%1000 == 0 and count > 0:
                #     pass
                #     store_elo_dict(elo_dict)
            except:
                traceback.print_exc()

    unrolled_elo_dicts = []
    for i, j in elo_dict.items():
        fighter_id = i[0]
        event_date = i[1]
        pre_elos = {'elo_pre_{0}_{1}'.format(k1, name):k2 for k1, k2 in j['pre'].items()}
        post_elos = {'elo_post_{0}_{1}'.format(k1, name):k2 for k1, k2 in j['post'].items()}

        temp_dict = dict()
        temp_dict.update({'fighter_id':fighter_id})
        temp_dict.update({'event_date': event_date})
        temp_dict.update(pre_elos)
        temp_dict.update(post_elos)
        unrolled_elo_dicts.append(temp_dict)

    df = pd.DataFrame.from_dict(unrolled_elo_dicts)
    return df


def fill_elos():
    with sqlite3.connect(db_location) as conn:
        df = pd.read_sql('select * from elo_table1', conn)

    # df = df[:1000]

    # df = df.set_index(['event_date', 'fighter_id'], drop = False)
    # df = df.sort_index()
    f_ids = set(df['fighter_id'])
    elo_cols = [i for i in df.columns if 'elo' in i]
    elo_cols.sort()

    dfs_f = []

    for f in tqdm.tqdm(f_ids):
        # print(f)
        f_df = df[df['fighter_id'] == f]
        f_df = f_df.sort_values('event_date')

        f_df = f_df.set_index('event_date', drop=False)

        last_found_elo = {i:starting_elo for i in elo_cols}
        for e_d in f_df['event_date']:
            for c in elo_cols:
                # print(f, c)
                # print(f, e_d, type(f_df.loc[e_d, c]), f_df.loc[e_d, c])

                if type(f_df.loc[e_d, c]) == np.float64:
                    f_e_elo = f_df.loc[e_d, c]
                else:
                    f_e_elo = f_df.loc[e_d, c].tolist()[0]

                if np.isnan(f_e_elo):
                    if 'post' in c:
                        # df.loc[(e_d, f), c] = last_found_elo[c]
                        f_df.loc[e_d, c] = last_found_elo[c]
                    else:
                        # df.loc[(e_d, f), c] = last_found_elo[c.replace('pre', 'post')]
                        f_df.loc[e_d, c] = last_found_elo[c.replace('pre', 'post')]
                else:
                    last_found_elo[c] = f_e_elo

        # f_df = f_df.dr
        # print(f_df.shape)
        # if f_df.shape[0] > 5:
        #     print('here')
        dfs_f.append(f_df)

    df = pd.concat(dfs_f)

    # df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(path + '/elo_table2.csv', index = False)
    # df = pd.read_csv(path + '/elo_table2.csv')
    # df = df.reset_index(drop=True)
    # print(df.columns)
    # print(df.index)
    #
    #
    # with sqlite3.connect(db_location) as conn:
    #     try:
    #         conn.execute('Drop table elo_table2')
    #     except:
    #         traceback.print_exc()
    #     df.to_sql('elo_table2', conn)

    # return df



def calculate_elos():
    df = load_raw_df()
    all_elo_df = calculate_elo_df(df, name='all')

    ko_df = df[df['standardized_method'] == 'KO']
    ko_df = calculate_elo_df(ko_df, name='KO')

    dec_df = df[df['standardized_method'] == 'Decision']
    dec_df = calculate_elo_df(dec_df, name='Decision')

    sub_df = df[df['standardized_method'] == 'Submission']
    sub_df = calculate_elo_df(sub_df, name='Submission')

    other_df = df[df['standardized_method'] == 'Other']
    other_df = calculate_elo_df(other_df, name='Other')

    df = df.merge(all_elo_df, on = ['fighter_id', 'event_date'], how = 'outer')
    df = df.merge(ko_df, on=['fighter_id', 'event_date'], how = 'outer')
    df = df.merge(dec_df, on=['fighter_id', 'event_date'], how = 'outer')
    df = df.merge(sub_df, on=['fighter_id', 'event_date'], how = 'outer')
    df = df.merge(other_df, on=['fighter_id', 'event_date'], how = 'outer')

    df.to_csv(path + 'all_elo_df.csv', index = False)

    with sqlite3.connect(db_location) as conn:
        try:
            conn.execute('Drop table elo_table1')
        except:
            traceback.print_exc()
        df.to_sql('elo_table1', conn)

    # df = fill_elos(df)
    # # df = df[df['elo_pre_{0}_all'.format(1000)] != starting_elo]
    #
    #
    # with sqlite3.connect(db_location) as conn:
    #     try:
    #         conn.execute('Drop table elo_table')
    #     except:
    #         traceback.print_exc()
    #     df.to_sql('elo_table', conn)



def load_raw_df():
    with sqlite3.connect(db_location) as conn:
        df = pd.read_sql(sql = 'Select * from matches', con=conn)
        # df = df[:10000]
    # print(len(set(df['fighter_id'])))

    df['standardized_method'] = np.nan
    df['standardized_method'] = df.apply(lambda x: 'Submission' if ('choke' in str(x['method'].lower()) or
    'bar' in str(x['method'].lower()) or 'lock' in str(x['method'].lower())) else x['standardized_method'], axis = 1)
    df['standardized_method'] = df.apply(lambda x: 'KO' if ('kick' in str(x['method'].lower()) or
    'punch' in str(x['method'].lower()) or 'elbow' in str(x['method'].lower()) or 'knee' in str(x['method'].lower())
    or 'pound' in str(x['method'].lower())) else x['standardized_method'], axis = 1)

    df['standardized_method'] = df.apply(lambda x: 'Decision' if ('decision' in str(x['method'].lower()))
    else x['standardized_method'], axis = 1)
    df['standardized_method'] = df['standardized_method'].fillna('Other')


    return df


def get_general_fighter_features(s, name):
    import datetime

    event_time_diff = s['event_time_diff']

    try:
        dob = s['dob']
        dob_dt = datetime.date(int(dob.split('.')[0]), int(dob.split('.')[1]), int(dob.split('.')[2]))

        event_date = s['event_date'].split(' ')[0]
        event_date_dt = datetime.date(int(event_date.split('-')[0]), int(event_date.split('-')[1]), int(event_date.split('-')[2]))

        age = (event_date_dt - dob_dt).days
        year = event_date_dt.year
        dob_month =dob_dt.month
        dob_year =dob_dt.year

    except:
        age = fill_value
        year = fill_value
        dob_month =fill_value
        dob_year =fill_value

        # traceback.print_exc()
    # print(year)
    # print(dob_month, dob_year)

    try:
        record = str(s['f_record']).replace('(','')
        wins = int(record.split('-')[0])
        loss = int(record.split('-')[1])
        other = int(record.split('-')[2])
        total_fights = wins + loss + other

        win_rate = wins/max(1, total_fights)
        other_rate = other / max(1, total_fights)

    except:
        wins = fill_value
        loss = fill_value
        other = fill_value
        total_fights = fill_value
        win_rate = fill_value
        other_rate = fill_value

    try:
        fight_weight = float(s['fight_weight'])*pound_conversion
        weight_allowance = float(s['fight_division']) - fight_weight
        division = float(s['fight_division'])
    except:
        fight_weight = fill_value
        weight_allowance = fill_value
        try:
            division =  float(str(s['fight_division']).replace('(', ''))
        except:
            division = fill_value

    try:
        height = float(s['height'])
        reach = float(s['reach'])
    except:
        height = fill_value
        reach = fill_value

    try:
        height_reach_ratio = height / reach
    except:
        height_reach_ratio = 0

    elo_cols = [i for i in s.index if 'elo' in i and 'pre' in i]
    elo_cols = list(set(elo_cols))
    elo_type_cols = [i + '_{0}'.format(name) for i in elo_cols]
    elo_dict = dict()

    for i, j in zip(elo_type_cols, elo_cols):
        elo_dict.update({i: s[j]})

    output_dict = {'age_{0}'.format(name):age,
                   'wins_{0}'.format(name):wins,
                   'loss_{0}'.format(name):loss,
                   'other_{0}'.format(name):other,
                   'total_fights_{0}'.format(name):total_fights,
                   'win_rate_{0}'.format(name):win_rate,
                   'other_rate_{0}'.format(name):other_rate,
                   'weight_allowance_{0}'.format(name):weight_allowance,
                   'fight_weight_{0}'.format(name):fight_weight,
                   'height_reach_ratio_{0}'.format(name):height_reach_ratio,
                   'event_time_diff_{0}'.format(name):event_time_diff,
                   'division':division,
                   'year':year,
                   'dob_year_{0}'.format(name):dob_year,
                   'dob_month_{0}'.format(name):dob_month}
    output_dict.update(elo_dict)
    return output_dict


def add_time_features():
    # with sqlite3.connect(db_location) as conn:
    #     df = pd.read_sql('select * from elo_table2', conn)

    df = pd.read_csv(path + '/elo_table2.csv')

    # df = df[::-1]
    df['event_dt'] = pd.to_datetime(df['event_date'], errors='coerce')
    df['event_timestamp'] = df['event_dt'].values.astype(np.int64) // 10 ** 9
    df['event_time_diff'] = df.groupby('fighter_id')['event_timestamp'].diff()

    df.to_csv(path + '/elo_table3.csv', index = False)
    # with sqlite3.connect(db_location) as conn:
    #     try:
    #         conn.execute('Drop table elo_table3')
    #     except:
    #         traceback.print_exc()
    #     df.to_sql('elo_table3', conn)


def get_fight_features(df, reverse = False):
    if reverse:
        df = df[::-1]
    results = {}
    elo_cols = [i for i in df.columns if 'elo' in i and 'post' not in i]
    elo_results = []

    for count, (k, v) in enumerate(df.iterrows()):
        results.update(get_general_fighter_features(v, count))
        elo_results.append(v[elo_cols])
    elo_diff = elo_results[0].values - elo_results[1].values

    results.update({i + '_diff': j for i, j in zip(elo_cols, elo_diff)})
    # results['weight_allowance_diff'] = results['weight_allowance_0'] - results['weight_allowance_1']

    if df.head(1)['result'].tolist()[0] == 'Win':
        results['target'] = 1
    elif df.head(1)['result'].tolist()[0] == 'Loss':
        results['target'] = 0
    else:
        results['target'] = .5

    return results


def get_general_features(df):
    df['loss_div'] = df['loss_0'] / df['loss_1']
    df['wins_div'] = df['wins_0'] / df['wins_1']
    df['total_fights_div'] = df['total_fights_0'] / df['total_fights_1']
    df['age_diff'] = df['age_0'] - df['age_1']

    df['weight_diff'] = df['fight_weight_0'] - df['fight_weight_1']
    df['height_reach_ratio_diff'] = df['height_reach_ratio_0'] - df['height_reach_ratio_1']
    df['event_time_div'] = df['event_time_diff_0'] / df['event_time_diff_1']

    df['elo_weight'] = df['elo_pre_1000_all_0'] + df['elo_pre_1000_all_1']
    return df


def get_features():
    df = pd.read_csv(path + '/elo_table3.csv')

    elo_cols = [i for i in df.columns if 'elo' in i]

    # for i in elo_cols:
    #     df[i] = df[i].fillna(starting_elo)
    print(df.shape)
    df = df.dropna(subset=['fight_id'])
    print(df.shape)
    fights = set(df['fight_id'])

    feature_list = []
    for i in tqdm.tqdm(fights):
        f_df = df[df['fight_id'] == i]
        if f_df.shape[0] == 2:
            feature_list.append(get_fight_features(f_df))
            feature_list.append(get_fight_features(f_df, reverse=True))

    res_df = pd.DataFrame.from_dict(feature_list)
    res_df = res_df.replace(fill_value, np.nan)
    res_df = res_df.astype(np.float64)
    res_df = get_general_features(res_df)



    res_df.to_csv(path + '/features.csv', index = False)
    print(res_df.shape)


def train_model():
    res_df = pd.read_csv(path + '/features.csv')
    res_df['elo_weight'] = res_df['elo_pre_1000_all_0'] + res_df['elo_pre_1000_all_1']
    res_df = res_df.reindex_axis(sorted(res_df.columns), axis=1)
    x = res_df.drop('target', axis = 1)
    y = res_df['target']

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.01, random_state=1)

    lgtrain = lgb.Dataset(train_x, train_y, weight=train_x['elo_weight'])
    lgvalid = lgb.Dataset(val_x, val_y, weight=val_x['elo_weight'])

    model = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=max_iter,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=1000,
        verbose_eval=10
    )
    model.save_model(path + '/lgbmodel', num_iteration=model.best_iteration)

    fi = model.feature_importance(iteration=model.best_iteration, importance_type='gain')
    fi_dicts = [(i, j) for i, j in zip(x.columns, fi)]
    fi_dicts.sort(key=operator.itemgetter(1), reverse=True)
    print(fi_dicts)


def predict(f_id = None, o_id = None, event_date = None, update_dict1 = None, update_dict2 = None, model = None, df = None):
    # with sqlite3.connect(db_location) as conn:
    #     df = pd.read_sql('select * from elo_table', conn)

    df['event_dt'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df[df['event_dt'] < event_date]
    # print(event_date)
    event_dt = datetime.datetime.strptime(event_date, '%Y-%m-%d %H:%M:%S')


    df1 = df[df['fighter_id'] == f_id]
    df2 = df[df['fighter_id'] == o_id]

    df1 = df1.sort_values('event_date').tail(1)
    df2 = df2.sort_values('event_date').tail(1)
    elo_results = []
    elo_cols = [i for i in df.columns if 'elo' in i and 'pre' not in i]

    f1_time_diff = (event_dt - df1.iloc[0]['event_dt']).days
    for _, i in df1.iterrows():
        s1 = i.copy()
        s1['event_date'] = event_date
        s1['event_time_diff'] = f1_time_diff
        elo_results.append(i[elo_cols])

    f2_time_diff = (event_dt - df2.iloc[0]['event_dt']).days
    for _, i in df2.iterrows():
        s2 = i.copy()
        s2['event_date'] = event_date
        s2['event_time_diff'] = f2_time_diff
        elo_results.append(i[elo_cols])

    results = dict()

    for k, v in update_dict1.items():
        s1[k] = v

    for k, v in update_dict2.items():
        s2[k] = v

    s1 = get_general_fighter_features(s1, 0)
    s2 = get_general_fighter_features(s2, 1)
    results.update(s1)
    results.update(s2)

    elo_diff = elo_results[0].values - elo_results[1].values
    results.update({i.replace('post', 'pre') + '_diff': j for i, j in zip(elo_cols, elo_diff)})
    results.update(results)
    pred_df = pd.DataFrame.from_dict([results])
    pred_df = pred_df.replace(fill_value, np.nan)
    pred_df = pred_df.astype(np.float64)
    pred_df = get_general_features(pred_df)

    pred_df = pred_df.reindex_axis(sorted(pred_df.columns), axis=1)

    # model = lgb.Booster(model_file=path + '/lgbmodel')

    return model.predict(pred_df)


def analysis():
    df = pd.read_csv(path + '/elo_table3.csv')
    df = df.sort_values('elo_post_1000_KO')
    df = df.dropna(subset=['elo_post_1000_KO'])
    a = 1


def main():
    calculate_elos()
    fill_elos()
    add_time_features()
    get_features()
    train_model()
    # test_predictions()



if __name__ == '__main__':
    main()