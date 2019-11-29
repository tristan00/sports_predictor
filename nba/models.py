from nba.process_data import DataManager
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

def run_naive_model():
    dm = DataManager()
    x, y = dm.get_labeled_data()
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x_train, x_val, y_train, y_val = train_test_split(x, y)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print(accuracy_score(y_val, rf.predict(x_val)))


def conv1d_cell(convolutional_filters,
                convolutional_kernel_size,
                activation,
                pooling_algorithm,
                convolutional_pool_size,
                convolution_type,
                pooling_layers,
                conv_layers_per_pooling_layer,
                batchnorm,
                data_format,
                dropout,
                dropout_amount):
    def f(input):
        for i in range(pooling_layers):
            for _ in range(1, conv_layers_per_pooling_layer):
                input = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                               activation=activation, data_format=data_format)(input)
                if batchnorm:
                    input = layers.BatchNormalization()(input)
            if pooling_algorithm:
                input = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(input)
            if dropout:
                input = eval(dropout)(dropout_amount)(input)
        return input

    return f


def recurrent_cell(recurrent_layers, recurrent_type, recurrent_layers_width, return_sequences,
                   batchnorm, dropout, dropout_amount):
    def f(input):
        for i in range(recurrent_layers):
            input = eval(recurrent_type)(recurrent_layers_width, return_sequences=(i >= recurrent_layers - 1) or return_sequences)(input)

            if batchnorm:
                input = layers.BatchNormalization()(input)
            if dropout:
                input = eval(dropout)(dropout_amount)(input)
        return input

    return f



def get_nn_model(input_shape1,
                 input_shape2,
                 convolutional_kernel_size,
                 convolutional_pool_size,
                 pooling_algorithm,
                 convolution_type,
                 recurrent_type,
                 use_x1,
                 use_x2,
                 optimizer_algorithm,
                 conv_layers_per_pooling_layer,
                 batchnorm,
                 block_1_activations,
                 block_2_activations,
                 merge_activations,
                 top_block_activations,
                 block_1_dropout,
                 block_2_dropout,
                 merge_block_dropout,
                 top_block_dropout,
                 merge_block_width,
                 block_2_width,
                 top_block_width,
                 recurrent_layers_width,
                 block_1_dense_layers_width,
                 merge_block_layers,
                 block_2_layers,
                 convolutional_filters,
                 dense_top_layers,
                 convolutional_layers,
                 recurrent_layers,
                 block_1_dense_layers,
                 block_1_dropout_amount,
            block_2_dropout_amount,
            merge_block_dropout_amount,
            top_block_dropout_amount,
                 ):


    if use_x1:
        input_layer_1 = layers.Input(shape=input_shape1)
    if use_x2:
        input_layer_2 = layers.Input(shape=input_shape2, name = 'x2')

    if use_x1:

        if convolutional_layers:
            x_conv = conv1d_cell(convolutional_filters, convolutional_kernel_size, block_1_activations,
                                 pooling_algorithm,
                                 convolutional_pool_size, convolution_type, convolutional_layers,
                                 conv_layers_per_pooling_layer, batchnorm=batchnorm, data_format='channels_last',
                                 dropout=block_1_dropout, dropout_amount=block_1_dropout_amount)(input_layer_1)
            x_conv = layers.Flatten(name='convolution_cell')(x_conv)
            x = x_conv

        if recurrent_layers:
            x_rnn = recurrent_cell(recurrent_layers=recurrent_layers,
                                   recurrent_type=recurrent_type, recurrent_layers_width=recurrent_layers_width,
                                   return_sequences=True, batchnorm=batchnorm, dropout=block_1_dropout, dropout_amount=block_1_dropout_amount)(
                input_layer_1)

            x_rnn = layers.Flatten()(x_rnn)
            if not (convolutional_layers):
                x = x_rnn
            else:
                x = layers.Concatenate()([x, x_rnn])

        if block_1_dense_layers:
            x_dnn = layers.Flatten(name='flatten15')(input_layer_1)
            for i in range(block_1_dense_layers):
                x_dnn = layers.Dense(block_1_dense_layers_width, activation=block_1_activations)(x_dnn)
                if batchnorm:
                    x_dnn = layers.BatchNormalization()(x_dnn)
                if block_1_dropout:
                    x_dnn = eval(block_1_dropout)(block_1_dropout_amount)(x_dnn)

            if not (
                     convolutional_layers or recurrent_layers):
                x = x_dnn
            else:
                x = layers.Concatenate(name='main_block_1_dense_layers')([x, x_dnn])

        if merge_block_layers and (
                convolutional_layers or recurrent_layers or block_1_dense_layers):
            x = layers.Dense(merge_block_width, activation=merge_activations)(x)
            if merge_block_dropout:
                x = eval(merge_block_dropout)(merge_block_dropout_amount)(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)

    if use_x2:

        x2 = layers.Dense(block_2_width, activation=block_2_activations)(input_layer_2)
        if block_2_dropout:
                x2 = eval(block_2_dropout)(block_2_dropout_amount)(x2)

        if batchnorm:
            x2 = layers.BatchNormalization()(x2)

        for i in range(1, block_2_layers):
            x2 = layers.Dense(block_2_width, activation=block_2_activations, name = 'merge_block_{}'.format(i))(x2)
            if block_2_dropout:
                # x2 = layers.Dropout(.2, name = 'block_2_dropout_1')(x2)
                x2 = eval(block_2_dropout)(block_2_dropout_amount)(x2)
            if batchnorm:
                x2 = layers.BatchNormalization()(x2)

        if use_x1:
            x = layers.Concatenate()([x, x2])
        else:
            x = x2

    for i in range(dense_top_layers):

        x = layers.Dense(top_block_width, activation=top_block_activations)(x)

        if i < dense_top_layers - 1:
            if top_block_dropout:
                x = eval(top_block_dropout)(top_block_dropout_amount)(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)

    out = layers.Dense(2, activation='softmax')(x)

    if use_x1 and use_x2:
        model = models.Model(inputs=[input_layer_1, input_layer_2], outputs=out)
    elif use_x1:
        model = models.Model(inputs=input_layer_1, outputs=out)
    else:
        model = models.Model(inputs=input_layer_2, outputs=out)

    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def optimize_nn():
    layers_width = list(range(16, 256))
    layers_width_2 = list(range(16, 128))
    convolutional_kernel_size = list(range(1, 10))
    convolutional_pool_size = list(range(1, 5))
    number_of_layers = [0, 1, 2, 3]

    pooling_algorithm = ['layers.MaxPooling1D', None]
    convolution_type = ['layers.Conv1D', 'layers.SeparableConv1D', 'layers.LocallyConnected1D']
    recurrent_type = ['layers.SimpleRNN', 'layers.LSTM', 'layers.GRU']

    activations = ['relu', 'elu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'hard_sigmoid', 'exponential']
    dropout = [None, 'layers.GaussianNoise', 'layers.Dropout', 'layers.GaussianDropout',
               'layers.AlphaDropout']
    binary_list = [0, 1]

    rotate_history_data = [False, True]
    history_lengths = [64]
    # transpose_history_data = [False]

    optimizer_algorithm = ['optimizers.SGD()',
                           'optimizers.RMSprop()',
                           'optimizers.Adagrad()',
                           'optimizers.Adadelta()',
                           'optimizers.Adamax()',
                           'optimizers.Nadam()']



    encoding_types = ['dense_autoencoder']
    encoding_sizes = [8]

    # team_encoding_types = ['dense_autoencoder']
    # player_encoding_types = ['dense_autoencoder']

    keys = []

    DatasetKey = collections.namedtuple('data_set_key', 'history_length, team_encoding_size, team_encoding_type, player_encoding_1_size, player_encoding_2_size, player_encoding_type')

    dms = dict()
    for history_length in history_lengths:
        for encoding_type in encoding_types:
            for encoding_size in encoding_sizes:
                key = DatasetKey(history_length=history_length,
                           team_encoding_size=encoding_size,
                           team_encoding_type=encoding_type,
                           player_encoding_1_size=encoding_size,
                           player_encoding_2_size=encoding_size,
                           player_encoding_type=encoding_type)

                dm = DataManager(fill_nans=True, data_scaling=True)
                dm.build_timeseries(history_length=history_length, team_encoding_size=encoding_size,
                                    team_encoding_type=encoding_type,
                                    player_encoding_1_size=encoding_size,
                                    player_encoding_2_size=encoding_size,
                                    player_encoding_type=encoding_type)
                x1, x2, y, columns = dm.get_labeled_data(history_length=history_length, get_history_data=True, team_encoding_size=encoding_size,
                                    team_encoding_type=encoding_type,
                                    player_encoding_1_size=encoding_size,
                                    player_encoding_2_size=encoding_size,
                                    player_encoding_type=encoding_type)
                print('get_labeled_data output', history_length, x1.shape, x2.shape, y.shape, len(columns))
                del dm
                gc.collect()
                x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1, x2, y, random_state=1)
                x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_val, x2_val, y_val, train_size=.5,
                                                                                   random_state=1)
                dms[key] = {'x1_train': x1_train,
                               'x1_val': x1_val,
                               'x1_test': x1_test,
                               'x2_train': x2_train,
                               'x2_val': x2_val,
                               'x2_test': x2_test,
                               'y_train': y_train,
                               'y_val': y_val,
                               'y_test': y_test}
                keys.append(key)

    try:
        results = pd.read_csv(f'{data_path}/nn_architectures.csv').to_dict(orient='records')
    except:
        results = list()

    while True:
        choice_dict = {
            'convolutional_kernel_size': random.choice(convolutional_kernel_size),
            'convolutional_pool_size': random.choice(convolutional_pool_size),
            'pooling_algorithm': random.choice(pooling_algorithm),
            'convolution_type': random.choice(convolution_type),
            'recurrent_type': random.choice(recurrent_type),
            'use_x1': random.choice(binary_list),
            'use_x2': random.choice(binary_list),
            'optimizer_algorithm': random.choice(optimizer_algorithm),
            'conv_layers_per_pooling_layer': random.choice(number_of_layers),
            'batchnorm': random.choice(binary_list),
            'block_1_activations': random.choice(activations),
            'block_2_activations': random.choice(activations),
            'merge_activations': random.choice(activations),
            'top_block_activations': random.choice(activations),
            'block_1_dropout': random.choice(dropout),
            'block_2_dropout': random.choice(dropout),
            'merge_block_dropout': random.choice(dropout),
            'top_block_dropout': random.choice(dropout),
            'merge_block_width': random.choice(layers_width),
            'block_2_width': random.choice(layers_width),
            'top_block_width': random.choice(layers_width),
            'recurrent_layers_width': random.choice(layers_width),
            'block_1_dense_layers_width': random.choice(layers_width),
            'merge_block_layers': random.choice(binary_list),
            'block_2_layers': random.choice(number_of_layers),
            'convolutional_filters': random.choice(layers_width_2),
            'dense_top_layers': random.choice(number_of_layers),
            'convolutional_layers': random.choice(number_of_layers),
            'recurrent_layers': random.choice(number_of_layers),
            'block_1_dense_layers': random.choice(number_of_layers),
            'block_1_dropout_amount':random.random()*.8,
            'block_2_dropout_amount':random.random()*.8,
            'merge_block_dropout_amount':random.random()*.8,
            'top_block_dropout_amount':random.random()*.8
        }

        try:
            assert choice_dict['use_x1'] or choice_dict['use_x2']
            key = random.choice(keys)

            x1_train = dms[key]['x1_train']
            x1_val = dms[key]['x1_val']
            x1_test = dms[key]['x1_test']

            rotate_history_data_choice = random.choice(binary_list)
            print(x1_train.shape)
            if rotate_history_data_choice:
                x1_train = np.swapaxes(x1_train,1,2)
                x1_val = np.swapaxes(x1_val,1,2)
                x1_test = np.swapaxes(x1_test,1,2)

            choice_dict['input_shape1'] = x1_train.shape[1:]
            choice_dict['input_shape2'] = dms[key]['x2_train'].shape[1:]

            print(dms[key]['x1_train'].shape, x1_train.shape)

            model = get_nn_model(**choice_dict)
            choice_dict['history_length'] = key.history_length
            choice_dict['team_encoding_size'] = key.team_encoding_size
            choice_dict['team_encoding_type'] = key.team_encoding_type
            choice_dict['player_encoding_1_size'] = key.player_encoding_1_size
            choice_dict['player_encoding_type'] = key.player_encoding_type
            choice_dict['rotate_history_data'] = rotate_history_data_choice
            choice_dict['num_of_params'] = model.count_params()

            start_time = time.time()
            cb = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=2,
                                         verbose=0, mode='auto')
            mcp_save = callbacks.ModelCheckpoint('{}/test.h5'.format(data_path), save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

            if choice_dict['use_x1'] and choice_dict['use_x2']:
                model.fit([x1_train, dms[key]['x2_train']],
                          dms[key]['y_train'],
                          validation_data=([x1_val, dms[key]['x2_val']],
                                           dms[key]['y_val']),
                          callbacks=[cb, mcp_save], epochs=200, batch_size=128)
            elif choice_dict['use_x1']:
                model.fit(x1_train,
                          dms[key]['y_train'],
                          validation_data=(x1_val,
                                           dms[key]['y_val']),
                          callbacks=[cb, mcp_save], epochs=200, batch_size=128)
            elif choice_dict['use_x2']:
                model.fit(dms[key]['x2_train'],
                          dms[key]['y_train'],
                          validation_data=(dms[key]['x2_val'],
                                           dms[key]['y_val']),
                          callbacks=[cb, mcp_save], epochs=200, batch_size=128)

            choice_dict['model_training_time'] = time.time() - start_time
            del model
            gc.collect()
            model = models.load_model('{}/test.h5'.format(data_path))

            if choice_dict['use_x1'] and choice_dict['use_x2']:
                preds = model.predict([x1_test, dms[key]['x2_test']])
            elif choice_dict['use_x1']:
                preds = model.predict(x1_test)
            elif choice_dict['use_x2']:
                preds = model.predict(dms[key]['x2_test'])
            else:
                raise Exception('invalid setup')

            preds = np.rint(preds[:, 1]).astype(int)
            truth = dms[key]['y_test'][:, 1]

            score = accuracy_score(
                truth,
                preds)
            choice_dict['accuracy'] = score
            del model, preds, score, cb, mcp_save, x1_train, x1_test, x1_val
            gc.collect()

        except AssertionError:
            continue
        except:
            traceback.print_exc()

        results.append(choice_dict)
        pd.DataFrame.from_dict(results).to_csv(f'{data_path}/nn_architectures.csv', index=False)


if __name__ == '__main__':
    optimize_nn()
