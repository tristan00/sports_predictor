from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pydot
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import traceback
from sklearn.svm import SVC
from deslib.des.knora_e import KNORAE
import gc
import glob
import operator
from sklearn import tree
import catboost
import lightgbm
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
#import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from feature_extractor import get_fightlist_features_df
import configparser

config = configparser.ConfigParser()
config.read('properties.ini')
db_location = config.get('mma', 'db_location')
model_location = config.get('mma', 'model_location')


def report(results, n_top=100):
    res_df = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)


        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            res_df.append({'mean': results['mean_test_score'][candidate],
                           'std':results['std_test_score'][candidate],
                           'rank':i,
                           'Parameters':results['params'][candidate],
                           'param_num_leaves':results['param_num_leaves'][candidate],
                           'param_num_iterations':results['param_num_iterations'][candidate],
                           'param_min_data_in_leaf': results['param_min_data_in_leaf'][candidate],
                           'param_max_bin': results['param_max_bin'][candidate],
                           'param_learning_rate': results['param_learning_rate'][candidate],
                           'param_boosting_type': results['param_boosting_type'][candidate]})

    res_df = pd.DataFrame.from_dict(res_df)
    res_df.to_csv('gridsearch_results.csv', index = False)


def test_model(clf, x, y, input_parameter_dict, n_iter_search = 25):
    random_search = RandomizedSearchCV(clf, param_distributions=input_parameter_dict,
                                       n_iter=n_iter_search, verbose=3)

    start = time.time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)


def get_features():
    with open(model_location + 'stored_features_temp.plk', 'rb') as infile:
        return pickle.load(infile)


def get_feature_df():
    return pd.read_pickle(model_location + 'stored_features_df.plk')


def preprocess_data(data_list, test_size = .1):

    data_tuples = [(i['full_features'], i['result_features']) for i in data_list]
    x = np.vstack([i[0] for i in data_tuples])
    y = np.vstack([i[1] for i in data_tuples])

    min_max_scaler = preprocessing.MinMaxScaler([-1,1])
    x = min_max_scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=test_size)
    return x_train, x_test, y_train, y_test


def get_data_matrix():
    files = glob.glob(model_location +'training_sets/*.plk')
    random.shuffle(files)
    ms = []
    columns = None
    for count, i in enumerate(files):
        df = pd.read_pickle(i)
        df = df.sample(frac=1)
        columns = df.columns
        m = df.as_matrix()
        print(count, m.shape)
        ms.append(m)



    return np.vstack(ms), columns


def preprocess_prediction(i_df):
    results = i_df.as_matrix()
    f1_x = results[:, :int((results.shape[1] - 5) / 2)]
    f2_x = results[:, int((results.shape[1] - 5) / 2):-5]
    meta_x = results[:, -5:-2]
    y = results[:, [-2]]
    f1_sub_f2_x = np.subtract(f1_x, f2_x)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_div_f2_x = np.true_divide(f1_x, f2_x)
        # f1_div_f2_x[f1_div_f2_x == np.inf] = np.nan
        # f1_div_f2_x[f1_div_f2_x == -np.inf] = np.nan
        # inds = np.where(np.isnan(f1_div_f2_x))
        # f1_div_f2_x[inds] = np.take(0, inds[1])
        f1_div_f2_x = np.nan_to_num(f1_div_f2_x)

    x = np.hstack([f1_x, f2_x, f1_sub_f2_x, f1_div_f2_x, meta_x])
    return x

def preprocess_df(test_size=.01):
    results, columns = get_data_matrix()
    np.random.shuffle(results)

    columns1 = [i for i in columns[0:int((results.shape[1] -5)/2)]]
    columns2 = [i for i in columns[int((results.shape[1] -5)/2):-5]]
    columns3 = [i for i in columns[-5:-2]]
    columns4 = [i + '_div_' + j for i,j in zip(columns1,columns2)]
    columns5 = [i + '_sub_' + j for i, j in zip(columns1, columns2)]


    f1_x = results[:, :int((results.shape[1] -5)/2)]
    f2_x = results[:, int((results.shape[1] -5)/2):-5]
    meta_x = results[:, -5:-2]
    y = results[:, [-2]]
    f1_sub_f2_x = np.subtract(f1_x, f2_x)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_div_f2_x = np.true_divide(f1_x, f2_x)
        f1_div_f2_x[f1_div_f2_x == np.inf] = np.nan
        f1_div_f2_x[f1_div_f2_x == -np.inf] = np.nan
        # fcol_median = np.nanmedian(f1_div_f2_x, axis=0)
        # inds = np.where(np.isnan(f1_div_f2_x))
        # f1_div_f2_x[inds] = np.take(fcol_median, inds[1])

    x = np.hstack([f1_x, f2_x,f1_sub_f2_x, f1_div_f2_x, meta_x])
    x = np.nan_to_num(x)
    column_names = columns1 + columns2 + columns5 + columns4 + columns3
    training_size = int(test_size*results.shape[0])
    x1_test = x[0:training_size,:]
    y_test = y[0:training_size,:]
    x1_train = x[training_size:,:]
    y_train = y[training_size:,:]

    # scaler1 = preprocessing.QuantileTransformer(output_distribution='normal')
    # x1_train = scaler1.fit_transform(x1_train)
    # x1_test = scaler1.transform(x1_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    return x1_train, y_train, x1_test, y_test, column_names


def preprocess_predictions(data_df):
    results = data_df.as_matrix()
    results = results[-1,:]
    results = results[:-1]
    results = np.reshape(results, (1, -1))
    with open(model_location + 'scaler.plk', 'rb') as infile:
        min_max_scaler = pickle.load(infile)
        results = min_max_scaler.transform(results)
    return results


def preprocess_results(train_res, test_res):
    min_max_scaler = preprocessing.QuantileTransformer(output_distribution='normal')
    train_res = min_max_scaler.fit_transform(train_res)
    test_res = min_max_scaler.transform(test_res)
    return train_res, test_res


def test_accuracy(predictions, results):
    correct = 0
    total = 0
    for i, j in zip(predictions, results):
        if (i[0] > i[1] and np.round(j) == 1) or (i[0] < i[1] and np.round(j) == 0):
            correct += 1
        total+= 1
    accuracy = 1 - (correct/total)
    print('Accuracy: ', accuracy)


def run_predictions_fn125(clf):

    date = '2018-02-03 00:00:00'
    fights = [{'f1':'/fighter/Lyoto-Machida-7513', 'f2':'/fighter/Eryk-Anders-93407'},
              {'f1': '/fighter/Valentina-Shevchenko-45384', 'f2': '/fighter/Priscila-Cachoeira-227399'},
              {'f1': '/fighter/Desmond-Green-89993', 'f2': '/fighter/Michel-Prazeres-22218'},
              {'f1': '/fighter/Marcelo-Golm-190735', 'f2': '/fighter/Timothy-Johnson-72706'},
              {'f1': '/fighter/Thiago-Santos-90021', 'f2': '/fighter/Anthony-Smith-29470'},
              {'f1': '/fighter/Tim-Means-11281', 'f2': '/fighter/Sergio-Moraes-21343'},
              {'f1': '/fighter/Alan-Patrick-Silva-Alves-31096', 'f2': '/fighter/Damir-Hadzovic-56139'},
              {'f1': '/fighter/Douglas-Silva-de-Andrade-87981', 'f2': '/fighter/Marlon-Vera-97179'},
              {'f1': '/fighter/Joe-Soto-17004', 'f2': '/fighter/Iuri-Alcantara-16129'},
              {'f1': '/fighter/Deiveson-Figueiredo-110485', 'f2': '/fighter/Joseph-Morales-123553'},
              {'f1': '/fighter/Maia-KahaunaeleStevenson-91345', 'f2': '/fighter/Eryk-Anders-93407'},
              {'f1': '/fighter/Lyoto-Machida-7513', 'f2': '/fighter/Polyana-Viana-Mota-153951'}]


    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        res['prediction'] = clf.predict_proba(i['result'].as_matrix())
        print(i['f1'], i['f2'], clf.predict_proba(i['result'].as_matrix()), clf.predict(i['result'].as_matrix()))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_ufc217(clf):
    #testing obvious results
    date = '2017-11-04 00:00:00'
    fights = [{'f1': '/fighter/Rose-Namajunas-69083', 'f2': '/fighter/Joanna-Jedrzejczyk-101411'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Rosi-Sexton-5358'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Aisling-Daly-25245'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Kelly-Warren-43679'}]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        res['prediction'] = clf.predict_proba(i['result'].as_matrix())
        print(i['f1'], i['f2'], clf.predict_proba(i['result'].as_matrix()), clf.predict(i['result'].as_matrix()))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_ufc217_2(clf):
    #testing obvious results
    date = '2017-11-04 00:00:00'
    fights = [{'f1': '/fighter/Rose-Namajunas-69083', 'f2': '/fighter/Joanna-Jedrzejczyk-101411'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Rosi-Sexton-5358'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Aisling-Daly-25245'},
              {'f1': '/fighter/Joanna-Jedrzejczyk-101411', 'f2': '/fighter/Kelly-Warren-43679'}]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file2.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_ufc221(clf):
    #testing obvious results
    date = '2018-02-10 00:00:00'
    fights = [{'f1': '/fighter/Yoel-Romero-60762', 'f2': '/fighter/Luke-Rockhold-23345'},
              {'f1': '/fighter/Mark-Hunt-10668', 'f2': '/fighter/Curtis-Blaydes-172939'},
              {'f1': '/fighter/Cyril-Asker-94411', 'f2': '/fighter/Tai-Tuivasa-133745'},
              {'f1': '/fighter/Jake-Matthews-122139', 'f2': '/fighter/Jingliang-Li-26381'},
              {'f1': '/fighter/Tyson-Pedro-146831', 'f2': '/fighter/Saparbek-Safarov-76834'},
              {'f1': '/fighter/Damien-Brown-62243', 'f2': '/fighter/Dong-Hyun-Kim-21673'},
              {'f1': '/fighter/Israel-Adesanya-56374', 'f2': '/fighter/Rob-Wilkinson-86279'},
              {'f1': '/fighter/Jeremy-Kennedy-104645', 'f2': '/fighter/Alexander-Volkanovski-101527'},
              {'f1': '/fighter/Jussier-da-Silva-36939', 'f2': '/fighter/Ben-Nguyen-8183'},
              {'f1': '/fighter/Ross-Pearson-11884', 'f2': '/fighter/Mizuto-Hirota-12078'},
              {'f1': '/fighter/Teruto-Ishihara-78898', 'f2': '/fighter/Jose-Alberto-Quinonez-152627'},
              {'f1': '/fighter/Luke-Jumeau-52853', 'f2': '/fighter/Daichi-Abe-191861'}
              ]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_ufc_fn126(clf):
    #testing obvious results
    date = '2018-02-10 00:00:00'
    fights = [{'f1': '/fighter/Donald-Cerrone-15105', 'f2': '/fighter/Yancy-Medeiros-27738'},
              {'f1': '/fighter/Derrick-Lewis-59284', 'f2': '/fighter/Marcin-Tybura-86928'},
              {'f1': '/fighter/James-Vick-81956', 'f2': '/fighter/Francisco-Trinaldo-31103'},
              {'f1': '/fighter/Thiago-Alves-5998', 'f2': '/fighter/Curtis-Millender-127439'},
              {'f1': '/fighter/Sage-Northcutt-130911', 'f2': '/fighter/Thibault-Gouti-124975'},
              {'f1': '/fighter/Jared-Gordon-74057', 'f2': '/fighter/Carlos-Diego-Ferreira-26358'},
              {'f1': '/fighter/Geoff-Neal-72107', 'f2': '/fighter/Brian-Camozzi-101093'},
              {'f1': '/fighter/Joby-Sanchez-50239', 'f2': '/fighter/Roberto-Sanchez-83282'},
              {'f1': '/fighter/Sarah-Moras-61600', 'f2': '/fighter/Lucie-Pudilova-159569'},
              {'f1': '/fighter/Brandon-Davis-67782', 'f2': '/fighter/Steven-Peterson-60433'},
              {'f1': '/fighter/Joshua-Burkman-10003', 'f2': '/fighter/Alex-Morono-64894'},
              {'f1': '/fighter/Oskar-Piechota-77303', 'f2': '/fighter/Tim-Williams-55756'}
              ]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_ufc_222(clf):
    #testing obvious results
    date = '2018-03-03 00:00:00'
    fights = [{'f1': '/fighter/Jordan-Johnson-124405', 'f2': '/fighter/Adam-Milstead-60258'},
              {'f1': '/fighter/Bryan-Caraway-13791', 'f2': '/fighter/Cody-Stamann-61896'},
              {'f1': '/fighter/Zak-Ottow-100359', 'f2': '/fighter/Mike-Pyle-4577'},
              {'f1': '/fighter/Hector-Lombard-11292', 'f2': '/fighter/CB-Dollaway-22350'},
              {'f1': '/fighter/John-Dodson-11660', 'f2': '/fighter/Pedro-Munhoz-52407'},
              {'f1': '/fighter/Beneil-Dariush-56583', 'f2': '/fighter/Alexander-Hernandez-97669'},
              {'f1': '/fighter/Mackenzie-Dern-137171', 'f2': '/fighter/Ashley-Yoder-392983'},
              {'f1': '/fighter/Cat-Zingano-33932', 'f2': '/fighter/Ketlen-Vieira-178961'},
              {'f1': '/fighter/Stefan-Struve-15063', 'f2': '/fighter/Andrei-Arlovski-270'},
              {'f1': '/fighter/Sean-OMalley-135099', 'f2': '/fighter/Andre-Soukhamthath-67967'},
              {'f1': '/fighter/Frankie-Edgar-14204', 'f2': '/fighter/Brian-Ortega-65310'},
              {'f1': '/fighter/Cristiane-Justino-14477', 'f2': '/fighter/Yana-Kunitskaya-49412'}
              ]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def run_predictions_fn_127(clf):
    #testing obvious results
    date = '2018-03-03 00:00:00'
    fights = [{'f1': '/fighter/Nasrat-Haqparast-122581', 'f2': '/fighter/Nad-Narimani-67643'},
              {'f1': '/fighter/Mark-Godbeer-54637', 'f2': '/fighter/Dmitriy-Sosnovskiy-116389'},
              {'f1': '/fighter/Steven-Ray-59838', 'f2': '/fighter/Kajan-Johnson-5615'},
              {'f1': '/fighter/Paul-Craig-110167', 'f2': '/fighter/Magomed-Ankalaev-170785'},
              {'f1': '/fighter/Hakeem-Dawodu-158725', 'f2': '/fighter/Danny-Henry-59830'},
              {'f1': '/fighter/Danny-Roberts-64677', 'f2': '/fighter/Oliver-Enkamp-122281'},
              {'f1': '/fighter/Charles-Byrd-51234', 'f2': '/fighter/John-Phillips-13470'},
              {'f1': '/fighter/Leon-Edwards-62665', 'f2': '/fighter/Peter-Sobotta-15816'},
              {'f1': '/fighter/Tom-Duquesnoy-92239', 'f2': '/fighter/Terrion-Ware-106755'},
              {'f1': '/fighter/Jimi-Manuwa-37528', 'f2': '/fighter/Jan-Blachowicz-25821'},
              {'f1': '/fighter/Fabricio-Werdum-8390', 'f2': '/fighter/Alexander-Volkov-40951'}
              ]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file.csv', 'w') as f:
        output_df.to_csv(f, index=False)

def run_predictions_ufc223(clf):
    #testing obvious results
    date = '2018-03-03 00:00:00'
    fights = [{'f1': '/fighter/Khabib-Nurmagomedov-56035', 'f2': '/fighter/Al-Iaquinta-42817'},
              {'f1': '/fighter/Rose-Namajunas-69083', 'f2': '/fighter/Joanna-Jedrzejczyk-101411'},
              {'f1': '/fighter/Renato-Carneiro-61700', 'f2': '/fighter/Calvin-Kattar-23782'},
              {'f1': '/fighter/Karolina-Kowalkiewicz-101401', 'f2': '/fighter/Felice-Herrig-42432'},
              {'f1': '/fighter/Joe-Lauzon-4923', 'f2': '/fighter/Chris-Gruetzemacher-36924'},
              {'f1': '/fighter/Olivier-AubinMercier-86241', 'f2': '/fighter/Evan-Dunham-22038'},
              {'f1': '/fighter/Zabit-Magomedsharipov-114261', 'f2': '/fighter/Kyle-Bochniak-86246'},
              {'f1': '/fighter/Bec-Rawlings-84964', 'f2': '/fighter/Ashlee-EvansSmith-75021'},
              {'f1': '/fighter/Devin-Clark-72777', 'f2': '/fighter/Mike-Rodriguez-107891'}
              ]
    features = get_fightlist_features_df(fights, date)
    print()

    results =  []
    for i in features:
        res = i.copy()
        prediction_array = preprocess_prediction(res['result'])
        res['prediction'] = clf.predict_proba(prediction_array)
        print(i['f1'], i['f2'], clf.predict_proba(prediction_array), clf.predict(prediction_array))
        results.append(res)
    output_df = pd.DataFrame.from_dict(results)
    with open('output_file2.csv', 'w') as f:
        output_df.to_csv(f, index=False)


def test_models():
    x1_train, y_train, x1_test, y_test = preprocess_df()

    #ada
    print('tuning adaboost')
    ada_parameters = {"algorithm" : ["SAMME", "SAMME.R"],
              "n_estimators": [50, 100, 500]
             }
    clf = AdaBoostClassifier()
    test_model(clf, x1_train, y_train, ada_parameters, n_iter_search=6)

    #gradient boosting
    print('tuning gradient boost')
    gb_parameters = {"loss" : ["deviance", "exponential"],
              "learning_rate " :   [.01, .1, .5],
              "n_estimators": [50, 100, 500],
            'max_depth':[2,3,5],
            'criterion':['friedman_mse', 'mse', 'mae'],
            'subsample':[.5, 1],
            'max_features':['sqrt', None]}
    clf = GradientBoostingClassifier()
    test_model(clf, x1_train, y_train, gb_parameters)

    #rf
    print('tuning random forest')
    rf_parameters = {"criterion" : ["gini", "entropy"],
              "n_estimators": [50, 100, 500],
            'max_depth':[5, 10, None],
            'max_features':['sqrt', None]}
    clf = RandomForestClassifier()
    test_model(clf, x1_train, y_train, rf_parameters)


    #ef
    print('tuning extra random forest')
    et_parameters = {"n_estimators": [50, 100, 500, 1000],
            'max_depth':[10, None],
            'max_features':['sqrt', None]}
    clf = ExtraTreesClassifier()
    test_model(clf, x1_train, y_train, et_parameters)



def feature_anayzer():
    x1_train, y_train, x1_test, y_test, column_names = preprocess_df()

    clf = RandomForestClassifier(n_estimators=200, max_features='sqrt')
    clf.fit(x1_train, y_train)
    feature_importance = clf.feature_importances_

    feature_dicts =[{'feature_name':i, 'importance':j} for i,j in zip(column_names, feature_importance)]
    df = pd.DataFrame.from_dict(feature_dicts)
    df.to_csv('test.csv')

def tree_builder():
    x1_train, y_train, x1_test, y_test, column_names = preprocess_df()
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(x1_train,y_train)
    print(clf.score(x1_test, y_test))
    with open("dtree2.dot", 'w') as dotfile:
        tree.export_graphviz(clf, out_file=dotfile, feature_names=column_names)


def tune_catboost():
    clf = catboost.CatBoostClassifier()
    x1_train, y_train, x1_test, y_test, column_names = preprocess_df()

    input_parameter_dict = {'depth': [6, 7, 8, 9],
                            'bagging_temperature': [0, .2, .4, .6],
                            'l2_leaf_reg':[1, 3, 5, 10],
                            'learning_rate':[.03, .05, .1]}

    random_search = RandomizedSearchCV(clf, param_distributions=input_parameter_dict,
                                       n_iter=50, verbose=3)

    start = time.time()
    random_search.fit(x1_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), 50))
    report(random_search.cv_results_)


def tune_lgbm():
    clf = lightgbm.LGBMClassifier()
    x1_train, y_train, x1_test, y_test, column_names = preprocess_df()

    input_parameter_dict = {'num_leaves': [15, 31, 63, 128],
                            'min_data_in_leaf': [5, 10, 20, 50],
                            'learning_rate':[.05, .1, .2],
                            'max_bin':[255, 512, 1024],
                            'boosting_type':['gbdt', 'dart'],
                            'num_iterations': [100, 120]}

    random_search = RandomizedSearchCV(clf, param_distributions=input_parameter_dict,
                                       n_iter=100, verbose=3)

    start = time.time()
    random_search.fit(x1_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), 50))
    report(random_search.cv_results_)

def test_stacking():
    results, columns = get_data_matrix()
    np.random.shuffle(results)

    columns1 = [i for i in columns[0:int((results.shape[1] -5)/2)]]
    columns2 = [i for i in columns[int((results.shape[1] -5)/2):-5]]
    columns3 = [i for i in columns[-5:-2]]
    columns4 = [i + '_div_' + j for i,j in zip(columns1,columns2)]
    columns5 = [i + '_sub_' + j for i, j in zip(columns1, columns2)]


    f1_x = results[:, :int((results.shape[1] -5)/2)]
    f2_x = results[:, int((results.shape[1] -5)/2):-5]
    meta_x = results[:, -5:-2]
    y = results[:, [-2]]
    f1_sub_f2_x = np.subtract(f1_x, f2_x)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_div_f2_x = np.true_divide(f1_x, f2_x)
        f1_div_f2_x[f1_div_f2_x == np.inf] = np.nan
        f1_div_f2_x[f1_div_f2_x == -np.inf] = np.nan
        # fcol_median = np.nanmedian(f1_div_f2_x, axis=0)
        # inds = np.where(np.isnan(f1_div_f2_x))
        # f1_div_f2_x[inds] = np.take(fcol_median, inds[1])

    x = np.hstack([f1_x, f2_x,f1_sub_f2_x, f1_div_f2_x, meta_x])
    x = np.nan_to_num(x)

    del results, f1_sub_f2_x, f1_div_f2_x, meta_x

    x1, x2, y1, y2 = train_test_split(x, y, train_size = .5)

    x2, x3, y2, y3= train_test_split(x2, y2, train_size=.8)

    chunks = 10

    l1_models = []
    for i in range(chunks):
        clf = lightgbm.LGBMClassifier()
        temp_x = x1[i*(x1.shape[0]//chunks):(1 + i)*(x1.shape[0]//chunks)]
        temp_y = y1[i*(y1.shape[0]//chunks):(1 + i)*(y1.shape[0]//chunks)]

        print(i, temp_x.shape, temp_y.shape)
        clf.fit(temp_x, temp_y)
        l1_models.append(clf)

    results1 = []
    results2 = []
    for i in l1_models:
        results1.append(np.expand_dims(i.predict(x2), 1))
        results2.append(np.expand_dims(i.predict(x3), 1))
    results1 = np.hstack(results1)
    results2 = np.hstack(results2)

    model_2 = lightgbm.LGBMClassifier()
    model_2.fit(results1, y2)
    print(model_2.score(results2, y3))
    del results1, results2, x1, y1, y2, y3, x3, model_2, l1_models
    gc.collect()

    model_3 = lightgbm.LGBMClassifier()
    x4, x5, y4, y5 = train_test_split(x, y, train_size = .9)
    model_3.fit(x4, y4)
    print(model_3.score(x5, y5))



def main():
    print('layer1')
    x1_train, y_train, x1_test, y_test, column_names = preprocess_df()

    #TODO: set up parameter optimization

    clfs1 = []
    #clf = catboost.CatBoostClassifier()
    clf =  lightgbm.LGBMClassifier()
    clf.fit(x1_train,y_train)
    print(clf.score(x1_test,y_test))
    run_predictions_ufc217_2(clf)
    run_predictions_ufc223(clf)

    # pool_classifiers = RandomForestClassifier(n_estimators=250, max_features='sqrt', max_depth=20)
    # pool_classifiers.fit(x1_train, y_train)
    # knorae = KNORAE(pool_classifiers)
    # knorae.fit(x1_train, y_train)
    # print(knorae.score(x1_test,y_test))


    #clfs1.append(ExtraTreesClassifier(n_estimators=1000))
    # clfs1.append(catboost.CatBoostClassifier())
    # clfs1.append(AdaBoostClassifier(n_estimators=200))
    # clfs1.append(GradientBoostingClassifier(n_estimators=200))
    # clfs1.append(RandomForestClassifier(n_estimators=200))



    # run_predictions_fn125(clf)
    # run_predictions_ufc217(clf)



if __name__ == '__main__':
    tune_lgbm()