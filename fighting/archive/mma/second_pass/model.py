import sqlite3
import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime
import random
from elo import rate_1vs1
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import operator

#fighter data
starting_elo = 1000
date_fighter_map = dict()
f_dict = None
result_methods = []
result_method_details = []
max_num_of_result_details = 100
result_details_dict = {}
result_details_read = False


# forest parameters
# Parameters
num_steps = 500
num_classes = 2
num_features = None
num_trees = 10
max_nodes = 1000

#manages and prepares the fighter data
class Fighter():
    def __init__(self, fighter_id, name, dob):
        self.fight_info_dict = {}
        self.dob = dob
        self.elo = starting_elo
        self.name = name

    def get_info_relevant_for_features(self, fight_date):

        if 'N/A' in self.dob:
            dob_available = 0
            difference_in_years = 0
        else:
            dob_available = 1
            if isinstance(self.dob, str):
                dob_datetime = datetime.datetime.strptime(self.dob, '%Y-%m-%d').date()
            difference_in_years = relativedelta(fight_date, dob_datetime).years

        previous_fights = [j for i, j in self.fight_info_dict.items() if i < fight_date]
        won_fights = [i for i in previous_fights if i['result'] == 'win']
        lost_fights = [i for i in previous_fights if i['result'] == 'loss']

        win_rate = len(won_fights)/max(1, len(previous_fights))
        num_of_fights = len(previous_fights)

        #previous win details
        win_features = extract_past_match_features(won_fights)
        loss_features = extract_past_match_features(lost_fights)
        general_features = extract_past_match_features(previous_fights)

        previous_fight_tuples = [(i, j) for i, j in self.fight_info_dict.items() if i < fight_date]
        previous_fight_tuples.sort(key=lambda tup: tup[0])
        if len(previous_fight_tuples) > 0:
            months_since_last_fight = abs(relativedelta(previous_fight_tuples[0][0], fight_date).months)
        else:
            months_since_last_fight = 0

        #streaks
        sorted_fights = [i[1] for i in previous_fight_tuples]
        last_fight_features = extract_past_match_features(sorted_fights[0:1])
        past_2_fight_features = extract_past_match_features(sorted_fights[0:2])
        past_3_fight_features = extract_past_match_features(sorted_fights[0:3])
        past_4_fight_features = extract_past_match_features(sorted_fights[0:4])
        past_5_fight_features = extract_past_match_features(sorted_fights[0:5])

        results =OrderedDict()
        results['dob_available'] = dob_available
        results['difference_in_years'] = difference_in_years
        results['win_features'] = win_features
        results['loss_features'] = loss_features
        results['general_features'] = general_features
        results['elo'] = self.elo
        results['months_since_last_fight'] = months_since_last_fight
        results['last_fight_features'] = last_fight_features
        results['past_2_fight_features'] = past_2_fight_features
        results['past_3_fight_features'] = past_3_fight_features
        results['past_4_fight_features'] = past_4_fight_features
        results['past_5_fight_features'] = past_5_fight_features
        results['fight_year'] = fight_date.year
        return results


    #extracts fight data, returns date
    def read_fight(self, info_list):
        fight_dict = extract_fight_info(info_list[3:])
        fight_dict['opponent_id'] = info_list[1]
        fight_datetime = datetime.datetime.strptime(info_list[2], '%Y-%m-%d %H:%M:00')
        fight_date = fight_datetime.date()
        fight_dict['fight_month'] = fight_datetime.month
        fight_dict['fight_year'] = fight_datetime.year
        fight_dict['fight_day'] = fight_datetime.day
        self.fight_info_dict[fight_date] = fight_dict
        return fight_date

    #assumes result is 'win' or 'loss', assumes opponent elo is valid num
    def update_elo(self, fight_date):
        fight = self.fight_info_dict[fight_date]
        opponent = f_dict.get(fight['opponent_id'], None)
        if opponent is None:
            opponent_elo = starting_elo
        else:
            opponent_elo = opponent.elo
        if fight['result'].lower() == 'win':
            result = rate_1vs1(self.elo, opponent_elo)
            self.elo = result[0]
        elif fight['result'].lower() == 'loss':
            result = rate_1vs1(opponent_elo, self.elo)
            self.elo = result[1]

    def reset_elo(self):
        self.elo = starting_elo

def extract_past_match_features(matches):
    result_dict = OrderedDict()
    sorted_result_methods = sorted(result_methods)
    sorted_result_method_details = sorted(result_method_details)
    methods = {}
    for i in sorted_result_methods:
        num_of_wins_with_method = [j for j in matches if j['method'] == i]
        methods[i] = len(num_of_wins_with_method)/max(1, len(matches))
    method_details = {}
    for i in sorted_result_method_details:
        num_of_wins_with_method = [j for j in matches if j['method_detail'] == i]
        method_details[i] = len(num_of_wins_with_method)/max(1, len(matches))
    rounds = {}
    for i in range(1,6):
        num_of_wins_with_method = [j for j in matches if j['round_finished'] == i]
        rounds[i] = len(num_of_wins_with_method)/max(1, len(matches))

    win_rate = sum([1 for i in matches if i['result'] == 'win']) / max(sum([1 for _ in matches]), 1)

    result_dict['methods'] = methods
    result_dict['method_details'] = method_details
    result_dict['rounds'] = rounds
    result_dict['win_rate'] = win_rate
    result_dict['fight_count'] = len(matches)
    return result_dict


def extract_fight_info(fight_input):
    global result_methods
    global result_details_dict

    round_finished = fight_input[2]
    result = fight_input[0]
    if 'decision' in fight_input[1].lower():
        method = 'decision'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'technical submission' in fight_input[1].lower():
        method = 'technical submission'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'submission' in fight_input[1].lower():
        method = 'submission'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'tko' in fight_input[1].lower():
        method = 'tko'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'ko' in fight_input[1].lower():
        method = 'ko'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    elif 'nc' in fight_input[1].lower():
        method = 'nc'
        if '(' in fight_input[1] and ')' in fight_input[1].split('(')[1]:
            method_detail = fight_input[1].split('(')[1].split(')')[0].lower()
        else:
            method_detail = ''
    else:
        method = ''
        method_detail = ''

    if method not in result_methods:
        result_methods.append(method)
    result_details_dict.setdefault(method_detail, 0)
    result_details_dict[method_detail] += 1

    return {'round_finished': round_finished,
            'result':result,
            'method':method,
            'method_detail':method_detail}

def get_all_fighters():
    with sqlite3.connect('mma2.db') as conn:
        res  = conn.execute('''select * from fighter''').fetchall()
        fighter_dict = dict()
        for i in res:
            fighter_dict[i[0]] = Fighter(i[0], i[1], i[2])
    return fighter_dict

#reads fights updates fighter class
def update_fighter_data():
    global date_fighter_map
    global f_dict
    with sqlite3.connect('mma2.db') as conn:
        res  = conn.execute('''select * from matches''').fetchall()
        for i in res:
            if i[0] in f_dict.keys():
                fight_date = f_dict[i[0]].read_fight(i)
                date_fighter_map.setdefault(fight_date, []).append(i[0])

def get_fight_dict(fight_date):
    with sqlite3.connect('mma2.db') as conn:
        res  = conn.execute('''select * from matches where result like ? or result like ?''', ('win', 'loss')).fetchall()
        result_dict = {}
        for i in res:
            fight_datetime = datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:00').date()
            if fight_datetime < fight_date:
                result_dict.setdefault(fight_datetime, []).append((i[0], i[1], fight_datetime))

        return result_dict

def update_elo(elo_date = None, reset_elo = True, start_date = None):
    global f_dict

    if isinstance(elo_date, str):
        elo_date = datetime.datetime.strptime(elo_date, '%Y-%m-%d %H:%M:00').date()

    if reset_elo:
        for i, j in f_dict.items():
            j.reset_elo()
    dates = date_fighter_map.keys()
    sorted_dates = sorted(dates)
    for i in sorted_dates:
        if start_date is not None and i < start_date:
            continue
        if elo_date is not None and i >= elo_date:
            break
        for j in date_fighter_map[i]:
            f_dict[j].update_elo(i)

#extracts features for random forest

# get data from features, put it in a list, TODO: put in numpy
def get_features(fighter1, fighter2, fight_date, result_exists = True):
    global num_features
    if isinstance(fight_date, str):
        fight_date = datetime.datetime.strptime(fight_date, '%Y-%m-%d %H:%M:00').date()
    f1_features = fighter1.get_info_relevant_for_features(fight_date)
    f2_features = fighter2.get_info_relevant_for_features(fight_date)
    if result_exists:
        if fighter1.fight_info_dict[fight_date]['result'] == 'win':
            f1_result = 1
            f2_result = 0
        elif fighter2.fight_info_dict[fight_date]['result'] == 'win':
            f1_result = 0
            f2_result = 1
        else:
            return None
    else:
        f1_result = None
        f2_result = None

    f1_feature_list = []
    for _, i in f1_features.items():
        if isinstance(i, dict):
            for _, j in i.items():
                if isinstance(j, dict):
                    for _, k in j.items():
                        f1_feature_list.append(k)
                else:
                    f1_feature_list.append(j)
        else:
            f1_feature_list.append(i)
    f2_feature_list = []
    for _, i in f2_features.items():
        if isinstance(i, dict):
            for _, j in i.items():
                if isinstance(j, dict):
                    for _, k in j.items():
                        f2_feature_list.append(k)
                else:
                    f2_feature_list.append(j)
        else:
            f2_feature_list.append(i)
    num_features = len(f1_feature_list + f2_feature_list)

    return [f1_feature_list + f2_feature_list, [f1_result, f2_result]]

#runs through list of dates extracting features from fights while updating elo for all fighters
def get_data(fight_date, test_size = .01):
    fight_date_dict = get_fight_dict(fight_date)
    test_x=[]
    test_y=[]
    train_x=[]
    train_y=[]
    raw_results = []
    last_date  = None

    for i, j in f_dict.items():
        j.reset_elo()

    dates = date_fighter_map.keys()
    sorted_dates = sorted(dates)
    for i in sorted_dates:
        update_elo(i, reset_elo = False, start_date=last_date)
        last_date = i
        if i not in fight_date_dict.keys():
            continue

        for j in fight_date_dict[i]:
            if j[0] not in f_dict.keys() or j[1] not in f_dict.keys():
                continue
            try:
                temp = get_features(f_dict[j[0]], f_dict[j[1]], j[2])
            except:
                continue
            if temp is not None:
                raw_results.append(temp)
    random.shuffle(raw_results)
    test_set = raw_results[0:int(test_size*len(raw_results))]
    train_set = raw_results[int(test_size * len(raw_results)):]

    for i in test_set:
        test_x.append(i[0])
        test_y.append(i[1])
    for i in train_set:
        train_x.append(i[0])
        train_y.append(i[1])

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return test_x, test_y, train_x, train_y

def evaluate_predictions(predictions, results):
    correct = 0
    total = 0
    predictions = predictions.tolist()

    for i in range(len(results)):
        total += 1
        if (predictions[i][0] > predictions[i][1] and results[i][0] > results[i][1]) or \
                (predictions[i][0] < predictions[i][1] and results[i][0] < results[i][1]):
            correct += 1

    return correct/total

def run_rf(num_of_trees = 256, criterion = 'gini', max_depth = None, min_samples_split  = 2, min_samples_leaf = 1 , \
           bootstrap =True, oob_score =False, fight_date = None):
    if fight_date is None:
        fight_date = datetime.datetime.now().date()

    test_x, test_y, train_x, train_y = get_data(fight_date = fight_date)
    clf = RandomForestClassifier(n_jobs=1, n_estimators = num_of_trees, criterion = criterion, max_depth = max_depth, \
                                 min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf , bootstrap =bootstrap, oob_score=oob_score)
    clf.fit(train_x, train_y)

    pred = clf.predict(test_x)
    accuracy = evaluate_predictions(pred, test_y)
    #print('accuracy:', accuracy)
    return clf, accuracy

def initiate_globals():
    global f_dict
    global result_method_details
    f_dict = get_all_fighters()
    update_fighter_data()
    result_method_details = [i[0] for i in sorted(result_details_dict.items(), key=operator.itemgetter(1), reverse=True)[:max_num_of_result_details]]

def main():
    global f_dict
    f_dict = get_all_fighters()
    update_fighter_data()
    today = datetime.datetime.now().date()
    update_elo(today)

    fighters = [i[1] for i in f_dict.items()]
    fighters.sort(key=lambda x: x.elo, reverse=True)
    for j in fighters:
        print( j.name, j.elo)
    run_rf()

def train(fight_date = None):
    initiate_globals()
    clf, accuracy = run_rf(fight_date = fight_date)
    return clf, accuracy

def predict_prob(clf, p1_id, p2_id, date):
    features = get_features(p1_id, p2_id, date, result_exists = False)
    x = np.array(features[0])
    x = x.reshape(1, -1)
    predictions = clf.predict_proba(x)
    prediction_array = [predictions[0][0][0], predictions[0][0][1]]
    return prediction_array

def predict(clf, p1_id, p2_id, fight_date=None):
    if fight_date is None:
        fight_date = datetime.datetime.now().date()
    f1 = f_dict[p1_id]
    f2 = f_dict[p2_id]
    features = get_features(f1, f2, fight_date, result_exists = False)
    x = np.array(features[0])
    x = x.reshape(1, -1)
    predictions = clf.predict(x)
    prediction_array = [predictions[0][0], predictions[0][1]]
    return prediction_array

#reverses the predictions and averages the results
def run_predictions(c, f1_id, f2_id, fight_date = None):
    if fight_date is None:
        fight_date = datetime.datetime.now().date()

    f1 = f_dict[f1_id]
    f2 = f_dict[f2_id]

    result1 = predict_prob(c, f1, f2, fight_date)
    result2 = predict_prob(c, f2, f1, fight_date)
    return [(result1[1] + result2[0])/2, (result1[0] + result2[1])/2]

def test_accuracy_for_different_parameters():

    possible_tree_nums = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    possible_criterion = ['gini', 'entropy']
    possible_max_depth = [None, 2, 4, 6, 8, 10,12,14,16,18,20]
    possible_min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    possible_min_samples_leaf = [1,2, 3, 4, 5]
    possible_bootstrap = [True, False]
    possible_oob_score = [True, False]

    initiate_globals()
    log = OrderedDict()
    tests = 1000
    for i in range(tests):
        try:
            num_of_trees, criterion, max_depth, min_samples_split, min_samples_leaf, bootstrap, oob_score = random.choice(possible_tree_nums), random.choice(possible_criterion),random.choice(possible_max_depth), \
                                                   random.choice(possible_min_samples_split), random.choice(possible_min_samples_leaf), random.choice(possible_bootstrap), random.choice(possible_oob_score)
            _, accuracy = run_rf(num_of_trees=num_of_trees, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, oob_score=oob_score)
            log[(i, num_of_trees, criterion, max_depth, min_samples_split, min_samples_leaf, bootstrap, oob_score)] = accuracy
        except Exception as e:
            print(e)
        print()
        print('results:')
        for k, v in log.items():
            print(k, v)



if __name__ == '__main__':
    test_accuracy_for_different_parameters()
    '''
    c, _ = train()

    fighters = [i[1] for i in f_dict.items()]
    fighters.sort(key=lambda x: x.elo, reverse=True)
    for count, j in enumerate(fighters):
        print(count, j.name, j.elo)

    f1_id ='/fighter/Conor-McGregor-29688'
    f2_id = '/fighter/Max-Holloway-38671'

    print(predict(c, f1_id, f2_id, fight_date=datetime.datetime.now().date()))

    print(predict(c, f2_id, f1_id, fight_date=datetime.datetime.now().date()))'''
