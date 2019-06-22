import sqlite3
import pandas as pd
import pickle

model_location = r'C:/Users/trist/Documents/db_loc/mma/models/'

def load_elo_dict():
    try:
        with open(model_location + 'elo_dict.pkl', 'rb') as infile:
            return pickle.load(infile)
    except FileNotFoundError:
        return dict()

d = load_elo_dict()

res = []

for k, v in d.items():
    elo_row = {'f_id':k[0], 'date':k[1]}
    # print(v)
    # print(v['pre'])
    # print(v['post'])
    elo_row.update({'pre'+str(i):j for i,j in v['pre'].items()})
    elo_row.update({'post'+str(i):j for i,j in v['post'].items()})
    res.append(elo_row)
df = pd.DataFrame.from_dict(res)

#df = df.sort_values('post1000')
#df = df.loc[df['f_id'] == '/fighter/Anderson-Silva-1356']
print(df.head())

# look at silva vs leites fght