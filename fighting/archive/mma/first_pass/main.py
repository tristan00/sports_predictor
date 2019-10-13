import sqlite3
import pandas as pd
import datetime
from elo import rate_1vs1
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

#fighter data
starting_elo = 1000
date_fighter_map = dict()
f_dict = None
result_methods = []
result_method_details = []


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
        #age info

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

        previous_fight_tuples = [(i, j) for i, j in self.fight_info_dict.items() if i < fight_date]
        previous_fight_tuples.sort(key=lambda tup: tup[0])
        if len(previous_fight_tuples) > 0:
            months_since_last_fight = relativedelta(previous_fight_tuples[0][0], fight_date).month
        else:
            months_since_last_fight = 0

        return {'dob_available':dob_available,
                'difference_in_years':difference_in_years,
                'win_rate':win_rate,
                'num_of_fights':num_of_fights,
                'win_features':win_features,
                'loss_features':loss_features,
                'elo': self.elo,
                'months_since_last_fight':months_since_last_fight}


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
    result_dict = dict()
    sorted_result_methods = sorted(result_methods)
    sorted_result_method_details = sorted(result_method_details)
    methods = {}
    for i in sorted_result_methods:
        num_of_wins_with_method = [i for i in matches if i['method'] == i]
        methods[i] = len(num_of_wins_with_method)/max(1, len(matches))
    method_details = {}
    for i in sorted_result_method_details:
        num_of_wins_with_method = [i for i in matches if i['method_detail'] == i]
        method_details[i] = len(num_of_wins_with_method)/max(1, len(matches))
    rounds = {}
    for i in [1, 2, 3, 4, 5]:
        num_of_wins_with_method = [i for i in matches if i['round_finished'] == i]
        rounds[i] = len(num_of_wins_with_method)/max(1, len(matches))

    result_dict['methods'] = methods
    result_dict['method_details'] = method_details
    result_dict['rounds'] = rounds
    return result_dict


def extract_fight_info(fight_input):
    global result_methods
    global result_method_details

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
    if method_detail not in result_method_details:
        result_method_details.append(method_detail)

    return {'round_finished': round_finished,
            'result':result,
            'method':method,
            'method_detail':method_detail}

def get_all_fighters():
    with sqlite3.connect('mma.db') as conn:
        res  = conn.execute('''select * from fighter''').fetchall()
        fighter_dict = dict()
        for i in res:
            fighter_dict[i[0]] = Fighter(i[0], i[1], i[2])
    return fighter_dict

#reads fights updates fighter class
def update_fighter_data():
    global date_fighter_map
    global f_dict
    with sqlite3.connect('mma.db') as conn:
        res  = conn.execute('''select * from matches''').fetchall()
        for i in res:
            if i[0] in f_dict.keys():
                fight_date = f_dict[i[0]].read_fight(i)
                date_fighter_map.setdefault(fight_date, []).append(i[0])

def get_fight_dict():
    with sqlite3.connect('mma.db') as conn:
        res  = conn.execute('''select * from matches where result like ? or result like ?''', ('win', 'loss')).fetchall()
        result_dict = {}
        for i in res:
            fight_datetime = datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:00').date()
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
        elif fighter1.fight_info_dict[fight_date]['result'] == 'win':
            f1_result = 0
            f2_result = 1
        else:
            return None, None
    else:
        f1_result = None
        f2_result = None

    f1_feature_list = []
    for _, i in f1_features.items():
        if isinstance(i, dict):
            for _, j in i.items():
                f1_feature_list.append(j)
        else:
            f1_feature_list.append(i)
    f2_feature_list = []
    for _, i in f2_features.items():
        if isinstance(i, dict):
            for _, j in i.items():
                f2_feature_list.append(j)
        else:
            f2_feature_list.append(i)
    num_features = len(f1_feature_list + f2_feature_list)

    return f1_feature_list + f2_feature_list, [f1_result, f2_result]

#runs through list of dates extracting features from fights while updating elo for all fighters
def get_data():
    fight_date_dict = get_fight_dict()
    x = []
    y = []
    last_date  = None

    for i, j in f_dict.items():
        j.reset_elo()

    dates = date_fighter_map.keys()
    sorted_dates = sorted(dates)
    for i in sorted_dates:
        update_elo(i, reset_elo = False, start_date=last_date)
        last_date = i
        print(i)
        if i not in fight_date_dict.keys():
            continue

        for j in fight_date_dict[i]:
            if j[0] not in f_dict.keys() or j[1] not in f_dict.keys():
                continue
            temp_x, temp_y = get_features(f_dict[j[0]], f_dict[j[1]], j[2])
            if temp_x is not None:
                x.append(temp_x)
                y.append(temp_y)
    return x, y

def run_rf():
    x = tf.placeholder(tf.float32, shape=[None, num_features])
    y = tf.placeholder(tf.int32, shape=[None])
    batch_x, batch_y = get_data()

    print(num_classes, num_features, num_trees, max_nodes)
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    train_op = forest_graph.training_graph(x, y)
    loss_op = forest_graph.training_loss(x, y)

    infer_op = forest_graph.inference_graph(x)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_vars = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_vars)


    for i in range(1, num_steps + 1):
        _, l = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

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



if __name__ == '__main__':
    main()