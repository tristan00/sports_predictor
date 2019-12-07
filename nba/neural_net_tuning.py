from nba.data_pipeline import (load_all_feature_file,
                               process_raw_data,
                               process_general_features,
                               generate_time_series_features)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks, optimizers
from nba.common import data_path
import numpy as np
import random
import traceback
import pandas as pd
import gc
import time
import collections
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
import pickle
import uuid


def conv1d_cell(convolutional_filters,
                convolutional_kernel_size,
                pooling_algorithm,
                convolutional_pool_size,
                convolution_type,
                cell_layers,
                pooling_layers_per_layer,
                convolutional_layers_per_layer,
                batchnorm,
                dropout,
                dropout_amount,
                activation):
    def f(input):
        for i in range(cell_layers):
            for _ in range(convolutional_layers_per_layer):
                input = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                               activation=activation)(input)
                if batchnorm:
                    input = layers.BatchNormalization()(input)
            if pooling_algorithm:
                for _ in range(pooling_layers_per_layer):
                    input = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(input)
            if dropout:
                input = eval(dropout)(dropout_amount)(input)
        return input

    return f


def recurrent_cell(recurrent_layers,
                   recurrent_type,
                   recurrent_layers_width,
                   return_sequences,
                   batchnorm,
                   dropout,
                   dropout_amount,
                   bidirectional,
                   activation,
                   recurrent_activation):
    def f(input):
        for i in range(recurrent_layers):
            if bidirectional:
                input = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width, activation=activation,
                                                                  recurrent_activation=recurrent_activation,
                                                                  return_sequences=(
                                                                                               i < recurrent_layers - 1) or return_sequences))(
                    input)
            else:
                input = eval(recurrent_type)(recurrent_layers_width, activation=activation,
                                             recurrent_activation=recurrent_activation,
                                             return_sequences=(i < recurrent_layers - 1) or return_sequences)(input)

            if batchnorm:
                input = layers.BatchNormalization()(input)
            if dropout:
                input = eval(dropout)(dropout_amount)(input)
        return input

    return f


def get_rnn_model(input_shape,
                  number_of_dense_hidden_layers,
                  optimizer_algorithm,
                  dense_layers_batchnorm,
                  dense_layers_activations,
                  dense_layers_width,
                  dense_layers_dropout_amount,
                  dense_layers_dropout_type,
                  recurrent_layers_batchnorm,
                  recurrent_layers_dropout_amount,
                  recurrent_layers_dropout_type,
                  recurrent_layers_width,
                  number_of_recurrent_layers,
                  recurrent_type,
                  recurrent_layers_bidirectional,
                  include_dense_top,
                  recurrent_layers_activation,
                  recurrent_layers_recurrent_activation):
    input_layer = layers.Input(input_shape)

    x = recurrent_cell(recurrent_layers=number_of_recurrent_layers,
                       recurrent_type=recurrent_type,
                       recurrent_layers_width=recurrent_layers_width,
                       return_sequences=False,
                       batchnorm=recurrent_layers_batchnorm,
                       dropout=recurrent_layers_dropout_type,
                       dropout_amount=recurrent_layers_dropout_amount,
                       bidirectional=recurrent_layers_bidirectional,
                       activation=recurrent_layers_activation,
                       recurrent_activation=recurrent_layers_recurrent_activation)(input_layer)
    # x = layers.Flatten()(x)
    if include_dense_top:
        for i in range(number_of_dense_hidden_layers):
            x = layers.Dense(dense_layers_width, activation=dense_layers_activations)(x)
            if i < number_of_dense_hidden_layers - 1:
                if dense_layers_batchnorm:
                    x = layers.BatchNormalization()(x)
                if dense_layers_dropout_type:
                    x = eval(dense_layers_dropout_type)(dense_layers_dropout_amount)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def get_cnn_model(input_shape,
                  number_of_dense_hidden_layers,
                  optimizer_algorithm,
                  dense_layers_batchnorm,
                  dense_layers_activations,
                  dense_layers_width,
                  dense_layers_dropout_amount,
                  dense_layers_dropout_type,
                  convolutional_layers_batchnorm,
                  convolutional_layers_activation,
                  convolutional_layers_dropout_amount,
                  convolutional_layers_dropout_type,
                  convolutional_layers_kernel_size,
                  convolutional_layers_pool_size,
                  convolutional_layers_pooling_algorithm,
                  convolutional_layers_convolution_type,
                  number_of_convolutional_cell_layers,
                  convolutional_layers_number_of_filters,
                  convolutional_layers_per_convolutional_cell_layer,
                  pooling_layers_per_convolutional_cell_layer,
                  include_dense_top):
    input_layer = layers.Input(input_shape)

    x = conv1d_cell(convolutional_filters=convolutional_layers_number_of_filters,
                    convolutional_kernel_size=convolutional_layers_kernel_size,
                    pooling_algorithm=convolutional_layers_pooling_algorithm,
                    convolutional_pool_size=convolutional_layers_pool_size,
                    convolution_type=convolutional_layers_convolution_type,
                    cell_layers=number_of_convolutional_cell_layers,
                    pooling_layers_per_layer=pooling_layers_per_convolutional_cell_layer,
                    convolutional_layers_per_layer=convolutional_layers_per_convolutional_cell_layer,
                    batchnorm=convolutional_layers_batchnorm,
                    dropout=convolutional_layers_dropout_type,
                    dropout_amount=convolutional_layers_dropout_amount,
                    activation=convolutional_layers_activation)(input_layer)

    # for i in range(number_of_convolutional_cell_layers):
    #     for _ in range(convolutional_layers_per_convolutional_cell_layer):
    #         input_layer = layers.Conv1D(filters=convolutional_layers_number_of_filters, kernel_size=convolutional_layers_kernel_size,
    #                                        activation=convolutional_layers_activation)(input_layer)
    #         if convolutional_layers_batchnorm:
    #             input_layer = layers.BatchNormalization()(input_layer)
    #     for _ in range(pooling_layers_per_convolutional_cell_layer):
    #         input_layer = layers.AveragePooling1D(pool_size=convolutional_layers_pool_size)(input_layer)
    #     if convolutional_layers_dropout_type:
    #         input_layer = eval(convolutional_layers_dropout_type)(convolutional_layers_dropout_amount)(input_layer)

    x = layers.Flatten()(x)
    if include_dense_top:
        for i in range(number_of_dense_hidden_layers):
            x = layers.Dense(dense_layers_width, activation=dense_layers_activations)(x)
            if i < number_of_dense_hidden_layers - 1:
                if dense_layers_batchnorm:
                    x = layers.BatchNormalization()(x)
                if dense_layers_dropout_type:
                    x = eval(dense_layers_dropout_type)(dense_layers_dropout_amount)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def get_dnn_model(input_shape,
                  number_of_dense_hidden_layers,
                  optimizer_algorithm,
                  dense_layers_batchnorm,
                  dense_layers_activations,
                  dense_layers_width,
                  dense_layers_dropout_amount,
                  dense_layers_dropout_type):
    input_layer = layers.Input(input_shape)
    x = layers.Flatten()(input_layer)

    for i in range(number_of_dense_hidden_layers):
        x = layers.Dense(dense_layers_width, activation=dense_layers_activations)(x)
        if i < number_of_dense_hidden_layers - 1:
            if dense_layers_batchnorm:
                x = layers.BatchNormalization()(x)
            if dense_layers_dropout_type:
                x = eval(dense_layers_dropout_type)(dense_layers_dropout_amount)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def optimize_nn():
    run_id = str(uuid.uuid4().hex)

    number_of_filters = list(range(16, 256))
    convolutional_kernel_size = list(range(1, 8))
    convolutional_pool_size = list(range(1, 8))
    number_of_dense_layers = [1, 2, 3]
    number_of_convolutional_cell_layers = [1, 2]
    number_of_recurrent_cell_layers = [1, 2]
    number_of_convolutional_layers = [1, 2]

    dense_layers_width = list(range(16, 512))
    recurrent_layers_width = list(range(16, 64))

    pooling_algorithm = ['layers.MaxPooling1D', 'layers.AveragePooling1D', None]
    convolution_type = ['layers.Conv1D', 'layers.SeparableConv1D', 'layers.LocallyConnected1D']
    recurrent_type = ['layers.LSTM', 'layers.GRU']

    activations = [None, 'relu', 'elu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'hard_sigmoid',
                   'exponential']
    dropout = [None, 'layers.GaussianNoise', 'layers.Dropout', 'layers.GaussianDropout',
               'layers.AlphaDropout']
    binary_list = [0, 1]
    encoding_types = ['pca', 'dense_autoencoder']
    optimizer_algorithm = ['optimizers.SGD()',
                           'optimizers.RMSprop()',
                           'optimizers.Adagrad()',
                           'optimizers.Adadelta()',
                           'optimizers.Adamax()',
                           'optimizers.Nadam()']

    history_lengths = [4, 8, 16, 32, 64]
    general_feature_encoding_size = [None, 16, 128]
    scaled = [True, False]
    # process_raw_data(sample = False)
    # process_general_features(aggregation_windows = [1, 5, 20], encoding_sizes = general_feature_encoding_size)
    # generate_time_series_features(history_lengths)
    data = load_all_feature_file(history_lengths, general_feature_encoding_size)

    # x2 = data['general_features_scaled'].drop(['win', 'score_diff', 'key'], axis = 1)
    y = data['general_features_scaled']['win']

    keys = []
    dms = dict()
    for e in general_feature_encoding_size:
        if e:
            x2 = data[f'encoded_general_features_{e}']
        else:
            x2 = data['general_features_scaled'].drop(['win', 'score_diff', 'key'], axis=1)

        for history_length in history_lengths:
            for s in scaled:
                print('here', e, history_length, s, data[f'time_series_{history_length}_{s}'].shape, x2.shape, y.shape,
                      type(data[f'time_series_{history_length}_{s}']), type(x2), type(y))
                x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
                    data[f'time_series_{history_length}_{s}'], x2, y, random_state=1)
                x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_val, x2_val, y_val, train_size=.5,
                                                                                   random_state=1)
                print(x1_train.shape, x2_train.shape, y_train.shape)
                dms[(history_length, s, e)] = {'x1_train': x1_train,
                                               'x1_val': x1_val,
                                               'x1_test': x1_test,
                                               'x2_train': x2_train,
                                               'x2_val': x2_val,
                                               'x2_test': x2_test,
                                               'y_train': y_train,
                                               'y_val': y_val,
                                               'y_test': y_test}
                keys.append((history_length, s, e))

    results = list()

    while True:

        try:
            key = random.choice(keys)

            x1_train = dms[key]['x1_train']
            x1_val = dms[key]['x1_val']
            x1_test = dms[key]['x1_test']

            rotate_history_data_choice = random.choice(binary_list)
            print(x1_train.shape)
            if rotate_history_data_choice:
                x1_train = np.swapaxes(x1_train, 1, 2)
                x1_val = np.swapaxes(x1_val, 1, 2)
                x1_test = np.swapaxes(x1_test, 1, 2)

            model_type = random.choice(['dnn', 'cnn', 'rnn'])
            parameter_dict = dict()
            parameter_dict['input_shape'] = x1_train.shape[1:]

            print(dms[key]['x1_train'].shape, x1_train.shape)

            if model_type == 'dnn':
                parameter_dict.update({'number_of_dense_hidden_layers': random.choice(number_of_dense_layers),
                                       'optimizer_algorithm': random.choice(optimizer_algorithm),
                                       'dense_layers_batchnorm': random.choice(binary_list),
                                       'dense_layers_activations': random.choice(activations),
                                       'dense_layers_width': random.choice(dense_layers_width),
                                       'dense_layers_dropout_amount': random.random() * .8,
                                       'dense_layers_dropout_type': random.choice(dropout), })
                model = get_dnn_model(**parameter_dict)
            elif model_type == 'cnn':
                parameter_dict.update({'number_of_dense_hidden_layers': random.choice(number_of_dense_layers),
                                       'optimizer_algorithm': random.choice(optimizer_algorithm),
                                       'dense_layers_batchnorm': random.choice(binary_list),
                                       'dense_layers_activations': random.choice(activations),
                                       'dense_layers_width': random.choice(dense_layers_width),
                                       'dense_layers_dropout_amount': random.random() * .8,
                                       'dense_layers_dropout_type': random.choice(dropout),
                                       'convolutional_layers_batchnorm': random.choice(binary_list),
                                       'convolutional_layers_dropout_amount': random.random() * .8,
                                       'convolutional_layers_dropout_type': random.choice(dropout),
                                       'convolutional_layers_kernel_size': random.choice(convolutional_kernel_size),
                                       'convolutional_layers_pool_size': random.choice(convolutional_pool_size),
                                       'convolutional_layers_pooling_algorithm': random.choice(pooling_algorithm),
                                       'convolutional_layers_convolution_type': random.choice(convolution_type),
                                       'number_of_convolutional_cell_layers': random.choice(
                                           number_of_convolutional_cell_layers),
                                       'convolutional_layers_number_of_filters': random.choice(number_of_filters),
                                       'convolutional_layers_per_convolutional_cell_layer': random.choice(
                                           number_of_convolutional_layers),
                                       'pooling_layers_per_convolutional_cell_layer': random.choice(
                                           number_of_convolutional_layers),
                                       'convolutional_layers_activation': random.choice(activations),
                                       'include_dense_top': random.choice(binary_list)})
                model = get_cnn_model(**parameter_dict)
            elif model_type == 'rnn':
                parameter_dict.update({'number_of_dense_hidden_layers': random.choice(number_of_dense_layers),
                                       'optimizer_algorithm': random.choice(optimizer_algorithm),
                                       'dense_layers_batchnorm': random.choice(binary_list),
                                       'dense_layers_activations': random.choice(activations),
                                       'dense_layers_width': random.choice(dense_layers_width),
                                       'dense_layers_dropout_amount': random.random() * .8,
                                       'dense_layers_dropout_type': random.choice(dropout),
                                       'recurrent_layers_batchnorm': random.choice(binary_list),
                                       'recurrent_layers_dropout_amount': random.random() * .8,
                                       'recurrent_layers_dropout_type': random.choice(dropout),
                                       'recurrent_layers_width': random.choice(recurrent_layers_width),
                                       'number_of_recurrent_layers': random.choice(number_of_recurrent_cell_layers),
                                       'recurrent_type': random.choice(recurrent_type),
                                       'include_dense_top': random.choice(binary_list),
                                       'recurrent_layers_bidirectional': random.choice(binary_list),
                                       'recurrent_layers_activation': random.choice(activations),
                                       'recurrent_layers_recurrent_activation': random.choice(activations)})
                print(parameter_dict)
                model = get_rnn_model(**parameter_dict)
            else:
                raise NotImplemented

            parameter_dict['history_length'] = key[0]
            parameter_dict['timeseries_scaled'] = key[1]
            parameter_dict['general_feature_encoding_size'] = key[2]
            parameter_dict['rotate_history_data'] = rotate_history_data_choice
            parameter_dict['num_of_params'] = model.count_params()
            parameter_dict['model_type'] = model_type

            start_time = time.time()
            cb = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=0,
                                         verbose=0, mode='auto')
            # mcp_save = callbacks.ModelCheckpoint('{}/test.h5'.format(data_path), save_best_only=True,
            #                                      monitor='val_loss',
            #                                      verbose=1)

            print(model_type, x1_train.shape, x1_val.shape, parameter_dict)
            model.fit(x1_train,
                      dms[key]['y_train'],
                      validation_data=(x1_val,
                                       dms[key]['y_val']),
                      callbacks=[cb], epochs=200, batch_size=128)

            parameter_dict['model_training_time'] = time.time() - start_time
            preds = model.predict(x1_test)

            # preds = np.rint(preds[:, 1]).astype(int)
            # truth = dms[key]['y_test'][:, 1]
            preds = np.rint(preds[:, 0]).astype(int)
            truth = np.rint(dms[key]['y_test']).astype(int)

            score = accuracy_score(
                truth,
                preds)
            parameter_dict['accuracy'] = score
            del model, preds, score, cb, x1_train, x1_test, x1_val
            gc.collect()

        except AssertionError:
            continue
        except:
            traceback.print_exc()

        results.append(parameter_dict)
        pd.DataFrame.from_dict(results).to_csv(f'{data_path}/nn_architectures_{run_id}.csv', index=False)


if __name__ == '__main__':
    optimize_nn()
