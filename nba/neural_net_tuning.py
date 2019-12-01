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


def get_recurrent_autoencoder(input_shape,
                          bottleneck_size,
                          dense_layer_activations,
                          bottleneck_layer_activations,
                          target_activations):

    from keras import backend as K

    input_layer = layers.Input(shape=input_shape)
    encoder = layers.LSTM(128)(input_layer)
    encoder = layers.Dense(32, activation=dense_layer_activations)(encoder)
    encoder_out = layers.Dense(bottleneck_size, activation=bottleneck_layer_activations, name='bottleneck')(encoder)
    decoder = layers.Dense(32, activation=dense_layer_activations)(encoder_out)
    decoder = layers.Dense(128, activation=dense_layer_activations)(decoder)
    decoder = K.expand_dims(decoder, axis = -1)
    out = layers.LSTM(128, return_sequences=True)(decoder)

    autoencoder = models.Model(input_layer, out)
    encoder = models.Model(input_layer, encoder_out)
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return autoencoder, encoder


def get_dense_autoencoder(input_shape,
                          bottleneck_size,
                          dense_layer_activations,
                          bottleneck_layer_activations,
                          target_activations):
    input_layer = layers.Input(shape=input_shape)
    encoder = layers.Dense(256, activation=dense_layer_activations)(input_layer)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(256, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(256, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(256, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(128, activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(64, activation=dense_layer_activations)(encoder)
    encoder_out = layers.Dense(bottleneck_size, activation=bottleneck_layer_activations, name='bottleneck')(encoder)
    decoder = layers.Dense(64, activation=dense_layer_activations)(encoder_out)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(128, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(256, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(256, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(256, activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(256, activation=dense_layer_activations)(decoder)
    out = layers.Dense(input_shape[0], activation=target_activations)(decoder)

    autoencoder = models.Model(input_layer, out)
    encoder = models.Model(input_layer, encoder_out)
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return autoencoder, encoder


class Encoder:
    def __init__(self, encoder_type, encoder_dims, encoder_id, encoder_params):
        self.encoder_type = encoder_type
        self.encoder_dims = encoder_dims
        self.encoder_id = encoder_id
        self.encoder_params = encoder_params
        self.model = None
        self.scaler_dict = dict()
        self.file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}'
        self.scaler_file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}_scaler'

    def fit(self, x):
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        assert len(x.shape) == 2

        x_df = pd.DataFrame(data = x,
                            columns=list(range(x.shape[1])))


        if self.encoder_type == 'pca':
            for i in x_df.columns:
                scaler = StandardScaler()
                if (x_df[i].isna().sum() / x_df.shape[0]) == 1.0:
                    x_df[i] = x_df[i].fillna(0)
                x_df[i]= scaler.fit_transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
                self.scaler_dict[i] = scaler
            self.model = PCA(n_components=self.encoder_dims)
            self.model.fit(x_df)
        if self.encoder_type in ['dense_autoencoder', 'recurrent_autoencoder']:
            for i in x_df.columns:
                scaler = StandardScaler()
                if (x_df[i].isna().sum() / x_df.shape[0]) == 1.0:
                     x_df[i] = x_df[i].fillna(0)
                x_df[i]= scaler.fit_transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
                # x_df[i] *= 2
                # x_df[i] -= 1
                self.scaler_dict[i] = scaler

            if self.encoder_type == 'dense_autoencoder':
                self.autoencoder, self.encoder = get_dense_autoencoder(input_shape = (x_df.shape[1],),
                                                  bottleneck_size = self.encoder_dims,
                                                  dense_layer_activations = 'elu',
                                                  bottleneck_layer_activations = 'linear',
                                                  target_activations= 'linear')
            if self.encoder_type == 'recurrent_autoencoder':
                x = np.expand_dims(x_df.values, 2)
                self.autoencoder, self.encoder = get_recurrent_autoencoder(input_shape = (x_df.shape[1], 1),
                                                  bottleneck_size = self.encoder_dims,
                                                  dense_layer_activations = 'elu',
                                                  bottleneck_layer_activations = 'linear',
                                                  target_activations= 'linear')
            x_train, x_val = train_test_split(x_df, random_state=1)

            early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=1,
                                         verbose=0, mode='auto')
            self.autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), callbacks=[early_stopping],
                           epochs=200, batch_size=32)
            self.save()

            for layer in self.autoencoder.layers:
                print(layer.name)

            for layer in self.encoder.layers:
                print(layer.name)

            self.model_output = self.encoder.predict(x)
            print(self.model_output.shape)
            return self.model_output

    def transform(self, x):
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        assert len(x.shape) == 2

        x_df = pd.DataFrame(data = x,
                            columns=list(range(x.shape[1])))

        if self.encoder_type == 'pca':
            for i in x_df.columns:
                if (x_df[i].isna().sum() / x_df.shape[0]) == 1.0:
                     x_df[i] = x_df[i].fillna(0)
                x_df[i]= self.scaler_dict[i].transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
            return self.model.transform(x_df)
        if self.encoder_type == 'dense_autoencoder':

            for i in x_df.columns:
                if (x_df[i].isna().sum() / x_df.shape[0]) == 1.0:
                     x_df[i] = x_df[i].fillna(0)
                x_df[i]= self.scaler_dict[i].transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))

            preds = self.encoder.predict(x_df)
            return preds

    def save(self):
        if self.encoder_type == 'pca':
            with open('{file_path}.pkl'.format(file_path=self.file_path), 'wb') as f:
                pickle.dump(self.model, f)

        if self.encoder_type == 'dense_autoencoder':
            models.save_model(self.encoder, '{file_path}_encoder.pkl'.format(file_path=self.file_path))
            models.save_model(self.autoencoder, '{file_path}_autoencoder.pkl'.format(file_path=self.file_path))
        with open('{file_path}.pkl'.format(file_path=self.scaler_file_path), 'wb') as f:
            pickle.dump(self.scaler_dict, f)

    def load(self):
        if self.encoder_type == 'pca':
            with open('{file_path}.pkl'.format(file_path=self.file_path), 'rb') as f:
                self.model = pickle.load(f)
        if self.encoder_type == 'dense_autoencoder':
            self.encoder = models.load_model('{file_path}_encoder.pkl'.format(file_path=self.file_path))
            self.autoencoder = models.load_model('{file_path}_autoencoder.pkl'.format(file_path=self.file_path))
        with open('{file_path}.pkl'.format(file_path=self.scaler_file_path), 'rb') as f:
            self.scaler_dict = pickle.load(f)


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

    out = layers.Dense(1, activation='sigmoid')(x)

    if use_x1 and use_x2:
        model = models.Model(inputs=[input_layer_1, input_layer_2], outputs=out)
    elif use_x1:
        model = models.Model(inputs=input_layer_1, outputs=out)
    else:
        model = models.Model(inputs=input_layer_2, outputs=out)

    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='binary_crossentropy',
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

    optimizer_algorithm = ['optimizers.SGD()',
                           'optimizers.RMSprop()',
                           'optimizers.Adagrad()',
                           'optimizers.Adadelta()',
                           'optimizers.Adamax()',
                           'optimizers.Nadam()']

    history_lengths = [4, 8, 16, 32, 64]
    general_feature_encoding_size = [None]
    scaled = [True, False]
    # process_raw_data(sample = False)
    # process_general_features()
    # generate_time_series_features(history_lengths, use_standard_scaler=True)
    data = load_all_feature_file(history_lengths, general_feature_encoding_size)

    # x2 = data['general_features_scaled'].drop(['win', 'score_diff', 'key'], axis = 1)
    y = data['general_features_scaled']['win']

    keys = []
    dms = dict()
    for e in general_feature_encoding_size:
        if e:
            x2 = data[f'encoded_general_features_{e}']
        else:
            x2 = data['general_features_scaled'].drop(['win', 'score_diff', 'key'], axis = 1)

        for history_length in history_lengths:
            for s in scaled:
                    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(data[f'time_series_{history_length}_{s}'], x2, y, random_state=1)
                    x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_val, x2_val, y_val, train_size=.5,
                                                                                       random_state=1)
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
            choice_dict['history_length'] = key[0]
            choice_dict['timeseries_scaled'] = key[1]
            choice_dict['general_feature_encoding_size'] = key[2]
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

            # preds = np.rint(preds[:, 1]).astype(int)
            # truth = dms[key]['y_test'][:, 1]
            preds =  np.rint(preds[:, 0]).astype(int)
            truth = np.rint(dms[key]['y_test']).astype(int)

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
