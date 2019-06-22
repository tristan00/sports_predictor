from train import predict, path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import lightgbm as lgb


def generate_event_page(url, date_str):

    df = pd.read_csv(path + '/elo_table3.csv')
    model = lgb.Booster(model_file=path + '/lgbmodel')

    s = requests.Session()
    r = s.get(url)
    soup = BeautifulSoup(r.text)

    card = soup.find('ul', {'class':'fightCard'})
    fights = card.find_all('li', {'class':'fightCard'})

    inputs = []
    for i in fights:
        # fighters = i.find_all('div', {'class':'fightCardBout'})
        l1 = i.find('div',{'class':'fightCardFighterName left'}).find_all('a')[0]['href']
        l2 = i.find('div',{'class':'fightCardFighterName right'}).find_all('a')[0]['href']

        inputs.append({'f_id':l1, 'o_id':l2,'event_date':date_str,'update_dict1':{},'update_dict2':{}, 'df':df, 'model':model})

    return inputs



ufc_fn_135 = generate_event_page('https://www.tapology.com/fightcenter/events/53519-ufc-230', '2018-11-03 00:00:00')

for i in ufc_fn_135:
    try:
        pred = predict(**i)
        print(i['f_id'], i['o_id'], pred)
    except:
        pass
        # print('exception', i)
