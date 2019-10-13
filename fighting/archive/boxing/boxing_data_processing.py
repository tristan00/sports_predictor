import pandas as pd
from common import boxing_data_location


df_fights = pd.read_csv('{}/fights_2019_06_23.csv'.format(boxing_data_location), sep = '|')
df_fighters = pd.read_csv('{}/fighters_2019_06_23.csv'.format(boxing_data_location), sep = '|')
a = 1