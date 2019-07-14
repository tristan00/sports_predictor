import pandas as pd
from collections import Counter
from scipy import stats
import traceback
import numpy as np
from process_data import create_data_files
from common import (data_path)
from sklearn.ensemble import RandomForestClassifier


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
target_cols = ['stat_ast', 'stat_ast_pct','stat_blk', 'stat_blk_pct','stat_def_rtg','stat_drb','stat_drb_pct','stat_efg_pct','stat_fg',
                  'stat_fg3','stat_fg3_pct','stat_fg3a','stat_fg3a_per_fga_pct','stat_fg_pct','stat_fga','stat_ft','stat_ft_pct',
                  'stat_fta','stat_fta_per_fga_pct','stat_mp','stat_off_rtg','stat_orb','stat_orb_pct','stat_pf','stat_plus_minus',
                  'stat_pts','stat_stl','stat_stl_pct','stat_tov', 'stat_tov_pct', 'stat_trb','stat_trb_pct', 'stat_ts_pct',
                  'stat_usg_pct','win']

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
            home_dict[(i, y)] = c.most_common(1)[0][0]
    return home_dict

def assign_home(df):
    home_dict = find_team_home_loc(df)
    df['t1_is_home'] = df.apply(lambda x: 1 if home_dict[(x['team_tag'], x['year'])] == x['location'] else 0, axis = 1)
    return df

def get_past_n_game_aggregates(df, n):
    team_tags = set(df['team_tag'])

    for i in team_tags:
        team_data = df[df['team_tag'] == i]
        team_data = team_data.sort_values(['year', 'month', 'day'])
        team_data



# df = pd.read_csv(r'C:\Users\trist\Documents\nba_data\boxscore_details.csv', sep = '|')
# df = assign_home(df)
# get_past_n_game_aggregates(df, 10)


# rating_df = pd.read_csv(r'C:\Users\trist\Documents\nba_data\team_rating_data.csv', sep = '|')
def feature_search():
    rating_df = pd.read_csv(r'C:\Users\trist\Documents\nba_data\team_features.csv', sep = '|')

    results = list()
    for i in rating_df.columns:
        for j in rating_df.columns:
            if i not in target_cols and j not in target_cols and i != j and rating_df[i].dtype in numerics and rating_df[j].dtype in numerics:
                try:
                    new_col = '{}_sub_by_{}'.format(i, j)

                    new_df = pd.concat([rating_df[i], rating_df[j], rating_df['win']], axis = 1)
                    new_df[new_col] = new_df[i] - new_df[j]

                    new_df2 = new_df.dropna()
                    print(new_df.shape, new_df2.shape)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(new_df2[new_col], new_df2['win'])
                    results.append({'column':new_col,
                                    'slope':slope,
                                    'intercept':intercept,
                                    'r_value':r_value,
                                    'p_value':p_value,
                                    'std_err':std_err,
                                    'r2_value':r_value*r_value})
                except:
                    traceback.print_exc()
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.sort_values(by = 'r2_value')
    results_df.to_csv('{data_path}/feature_results.csv'.format(data_path=data_path), sep = '|', index = False)


def generate_features(df):
    for i in df.columns:
        for j in df.columns:
            if i not in target_cols and j not in target_cols and i != j and df[i].dtype in numerics and df[j].dtype in numerics:
                new_col = '{}_sub_by_{}'.format(i, j)
                df[new_col] = df[i] - df[j]
    return df

def feature_reduction(max_features = 100):
    rating_df = pd.read_csv(r'C:\Users\trist\Documents\nba_data\team_features.csv', sep = '|')
    rating_df = generate_features(rating_df)
    results = list()
    y =  rating_df['win']

    x_df = rating_df.select_dtypes(include=numerics)
    x_df = x_df.drop(target_cols, axis = 1)
    x_df = x_df.fillna(x_df.median())

    rf = RandomForestClassifier(max_depth=12, n_estimators=100)
    rf.fit(x_df, y)

    features = []
    for i, j in zip(x_df.columns, rf.feature_importances_):
        features.append((i, j))
    features = sorted(features, key = lambda x: x[1], reverse=True)
    print(features)



'''
observations of data ignoring opponent:
past n game win rate, r2 values of.056 for n = 100
past n game avg offensive offensive , r2 values of.037 for n = 100, how is this calculated?
past n game pregame rating, r2 values of.037 for n = 100, rating formula could use work if this is weaker than win rate
top non-advanced stat is field goal percentage
Past defensive rating and turnover stats are negatively related to wins, it makes sense as teams would need these more if under pressure
'''


if __name__ == '__main__':
    # create_data_files()
    # feature_search()
    feature_reduction()
