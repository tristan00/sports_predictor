import sqlite3
from elo import rate_1vs1
from collections import OrderedDict
import datetime

#class to efficiently handle data requests when multiple models will need the same data
#will memoirize relevant info
starting_elo = 1000
result_methods = []
result_method_details = []
max_num_of_result_details = 100
result_details_dict = {}

class DataManager():
    def __init__(self):

        self.f_dict = get_all_fighters()
        self.date_fighter_map = dict()

    def update_fighter_data(self):
        with sqlite3.connect('mma2.db') as conn:
            res  = conn.execute('''select * from matches''').fetchall()
            for i in res:
                if i[0] in self.f_dict.keys():
                    fight_date = self.f_dict[i[0]].read_fight(i)
                    self.date_fighter_map.setdefault(fight_date, []).append(i[0])

class Fighter():
    def __init__(self, fighter_id, name, dob):
        self.fight_info_dict = {}
        self.dob = dob
        self.elo_dict = dict()
        self.name = name

    def get_info_relevant_for_features(self, fight_date):

        if 'N/A' in self.dob:
            dob_available = 0
            difference_in_years = 0
        else:
            dob_available = 1
            if isinstance(self.dob, str):
                dob_datetime = datetime.datetime.strptime(self.dob, '%Y-%m-%d').date()
            difference_in_years = datetime.relativedelta(fight_date, dob_datetime).years

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
            months_since_last_fight = abs(datetime.relativedelta(previous_fight_tuples[0][0], fight_date).months)
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
    def update_and_return_elo(self, fight_date, opponent_id, f_dict):
        fight = self.fight_info_dict[fight_date]
        opponent = f_dict[opponent_id]
        opponent_elo = opponent.get_elo(fight_date, f_dict)
        if fight['result'].lower() == 'win':
            result = rate_1vs1(self.elo, opponent_elo)
            self.elo = result[0]
        elif fight['result'].lower() == 'loss':
            result = rate_1vs1(opponent_elo, self.elo)
            self.elo = result[1]

    def reset_elo(self):
        self.elo = starting_elo

    def calculate_elo_change():

    def get_elo(self, fight_date, f_dict):
        past_fight_dates = [i for i, j in self.fight_info_dict.items() if i < fight_date]
        if len(past_fight_dates) == 0:
            self.elo_dict[fight_date] = starting_elo
        past_fight_dates.sort(reverse = True)
        return self.elo_dict.setdefault(fight_date, update_elo)

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

def get_all_fighters():
    with sqlite3.connect('mma2.db') as conn:
        res  = conn.execute('''select * from fighter''').fetchall()
        fighter_dict = dict()
        for i in res:
            fighter_dict[i[0]] = Fighter(i[0], i[1], i[2])
    return fighter_dict

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
