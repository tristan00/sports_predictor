import model
import datetime
import sqlite3
import re

name_mapping = {'Georges St-Pierre': '/fighter/Georges-St-Pierre-3500',
                'Michael Bisping':'/fighter/Michael-Bisping-10196',
                'T.J. Dillashaw':'/fighter/TJ-Dillashaw-62507',
                'Cody Garbrandt':'/fighter/Cody-Garbrandt-50381',
                'Rose Namajunas':'/fighter/Rose-Namajunas-69083',
                'Joanna Jędrzejczyk':'/fighter/Joanna-Jedrzejczyk-101411',
                'Stephen Thompson':'/fighter/Stephen-Thompson-59608',
                'Jorge Masvidal':'/fighter/Jorge-Masvidal-7688',
                'Paulo Costa':'/fighter/Paulo-Henrique-Costa-147165',
                'Johny Hendricks':'/fighter/Johny-Hendricks-24539',
                'James Vick':'/fighter/James-Vick-81956',
                'Joseph Duffy':'/fighter/Joseph-Duffy-17052',
                'Mark Godbeer':'/fighter/Mark-Godbeer-54637',
                'Walt Harris':'/fighter/Walt-Harris-72046',
                'Ovince Saint Preux':'/fighter/Ovince-St-Preux-38842',
                'Corey Anderson':'/fighter/Corey-Anderson-171723',
                'Randy Brown':'/fighter/Randy-Brown-115641',
                'Mickey Gall':'/fighter/Mickey-Gall-160145',
                'Curtis Blaydes':'/fighter/Curtis-Blaydes-172939',
                'Oleksiy Oliynyk':'/fighter/Alexey-Oleynik-2027',
                'Ricardo Ramos':'/fighter/Ricardo-Lucas-Ramos-121143',
                'Aiemann Zahabi':'/fighter/Aiemann-Zahabi-121009',
                'Max Holloway':'/fighter/Max-Holloway-38671',
                'José Aldo':'/fighter/Jose-Aldo-11506',
                'Francis Ngannou':'/fighter/Francis-Ngannou-152341',
                'Alistair Overeem':'/fighter/Alistair-Overeem-461',
                'Henry Cejudo': '/fighter/Henry-Cejudo-125297',
                'Sergio Pettis':'/fighter/Sergio-Pettis-50987',
                'Eddie Alvarez':'/fighter/Eddie-Alvarez-9265',
                'Justin Gaethje':'/fighter/Justin-Gaethje-46648',
                'Tecia Torres':'/fighter/Tecia-Torres-85096',
                'Michelle Waterson':'/fighter/Michelle-Waterson-23091',
                'Paul Felder':'/fighter/Paul-Felder-68205',
                'Charles Oliveira':'/fighter/Charles-Oliveira-30300',
                'Cub Swanson':'/fighter/Cub-Swanson-11002',
                'Brian Ortega':'/fighter/Brian-Ortega-65310',
                'Jason Knight':'/fighter/Jason-Knight-44957',
                'Gabriel Benítez': '/fighter/Gabriel-Benitez-25733',
                'Marlon Moraes': '/fighter/Marlon-Moraes-30936',
                'Aljamain Sterling': '/fighter/Aljamain-Sterling-66313',}


def clean_data(input_data):
    if isinstance(input_data, str):
        input_data =  re.sub(r'[-/n.]',' ', input_data)
        return ' '.join(input_data.split())
    else:
        return input_data

def lookup_id(name):
    with sqlite3.connect('mma2.db') as conn:
        res = conn.execute('select fighter_id from fighter where fighter_name like ?', (clean_data(name).replace(' ', '%') ,)).fetchall()
        print(res)
        return res[0]

def test_ufc_217():
    #ufc 217 tests
    day_before_ufc_217 = datetime.date(2017, 11, 3)
    matchups = [['Georges St-Pierre', 'Michael Bisping'],
                ['T.J. Dillashaw', 'Cody Garbrandt'],
                ['Rose Namajunas', 'Joanna Jędrzejczyk'],
                ['Stephen Thompson', 'Jorge Masvidal'],
                ['Paulo Costa', 'Johny Hendricks'],
                ['James Vick', 'Joseph Duffy'],
                ['Mark Godbeer','Walt Harris'],
                ['Ovince Saint Preux', 'Corey Anderson'],
                #['Randy Brown', 'Mickey Gall'],
                ['Curtis Blaydes', 'Oleksiy Oliynyk'],
                #['Ricardo Ramos', 'Aiemann Zahabi']
                ]
    clf, accuracy = model.train(day_before_ufc_217)
    print('tested accuracy:', accuracy)
    for i in matchups:
        #f1_id = lookup_id(i[0])
        #f2_id = lookup_id(i[1])

        f1_id = name_mapping[i[0]]
        f2_id = name_mapping[i[1]]
        print('Match {0} vs {1}, predicted result:'.format(i[0], i[1]), model.run_predictions(clf, f1_id, f2_id, day_before_ufc_217))

def test_ufc_218():
    #ufc 218 tests
    day_before_ufc_218 = datetime.date(2017, 12, 1)
    matchups = [['Max Holloway', 'José Aldo'],
                ['Francis Ngannou', 'Alistair Overeem'],
                ['Henry Cejudo', 'Sergio Pettis'],
                ['Eddie Alvarez', 'Justin Gaethje'],
                ['Tecia Torres', 'Michelle Waterson'],
                ['Paul Felder', 'Charles Oliveira']]
    '''['Yancy Medeiros','Alex Oliveira'],
                ['David Teymur', 'Drakkar Klose'],
                ['Felice Herrig', 'Cortney Casey'],
                ['Amanda Cooper', 'Angela Magaña'],
                ['Abdul Razak Alhassan', 'Sabah Homasi'],
                ['Dominick Reyes', 'Jeremy Kimball'],
                ['Justin Willis', 'Allen Crowder']'''

    clf, accuracy = model.train(day_before_ufc_218)
    print('tested accuracy:', accuracy)
    for i in matchups:
        #f1_id = lookup_id(i[0])
        #f2_id = lookup_id(i[1])

        f1_id = name_mapping[i[0]]
        f2_id = name_mapping[i[1]]
        print('Match {0} vs {1}, predicted result:'.format(i[0], i[1]), model.run_predictions(clf, f1_id, f2_id, day_before_ufc_218))

def test_ufc_fn_123():
    day_before_ufc_fn_123 = datetime.date(2017, 12, 5)
    matchups = [['Cub Swanson', 'Brian Ortega'],
                ['Jason Knight', 'Gabriel Benítez'],
                ['Marlon Moraes', 'Aljamain Sterling']]

    clf, accuracy = model.train(day_before_ufc_fn_123)
    print('tested accuracy:', accuracy)
    for i in matchups:
        f1_id = name_mapping[i[0]]
        f2_id = name_mapping[i[1]]
        print('Match {0} vs {1}, predicted result:'.format(i[0], i[1]), model.run_predictions(clf, f1_id, f2_id, day_before_ufc_fn_123))

if __name__ == '__main__':
    test_ufc_217()
    test_ufc_218()
    test_ufc_fn_123()