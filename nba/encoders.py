
from sklearn import preprocessing, decomposition, model_selection
from keras import layers, models, callbacks, optimizers
from nba.common import data_path, timeit
import pandas as pd
import numpy as np
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
    encoder = layers.Dense(1024, activation=dense_layer_activations)(input_layer)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(max(512, bottleneck_size), activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(max(256, bottleneck_size), activation=dense_layer_activations)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dense(max(256, bottleneck_size), activation=dense_layer_activations)(encoder)
    encoder_out = layers.Dense(bottleneck_size, activation=bottleneck_layer_activations, name='bottleneck')(encoder)
    decoder = layers.Dense(max(256, bottleneck_size), activation=dense_layer_activations)(encoder_out)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(max(256, bottleneck_size), activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(max(512, bottleneck_size), activation=dense_layer_activations)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dense(1024, activation=dense_layer_activations)(decoder)
    out = layers.Dense(input_shape[0], activation=target_activations)(decoder)

    autoencoder = models.Model(input_layer, out)
    encoder = models.Model(input_layer, encoder_out)
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return autoencoder, encoder


class Encoder:
    def __init__(self, encoder_type, encoder_dims, encoder_id, encoder_params):
        print(f'Making instance of a {encoder_type} encoder with {encoder_dims} latent dims. Encoder id: {encoder_id}')

        self.encoder_type = encoder_type
        self.encoder_dims = encoder_dims
        self.encoder_id = encoder_id
        self.encoder_params = encoder_params
        self.model = None
        self.scaler_dict = dict()
        self.file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}'
        self.scaler_file_path = f'{data_path}/{encoder_type}_{encoder_dims}_{encoder_id}_scaler'

    @timeit
    def fit(self, x):
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        assert len(x.shape) == 2
        print(f'Encoder fit input shape: {x.shape}')

        x_df = pd.DataFrame(data = x,
                            columns=list(range(x.shape[1])))


        if self.encoder_type == 'pca':
            for i in x_df.columns:
                scaler = preprocessing.StandardScaler()
                if (x_df[i].isna().sum() / x_df.shape[0]) == 1.0:
                    x_df[i] = x_df[i].fillna(0)
                x_df[i]= scaler.fit_transform(x_df[i].fillna(x_df[i].median()).values.reshape(-1, 1))
                self.scaler_dict[i] = scaler
            self.model = decomposition.PCA(n_components=self.encoder_dims)
            self.model.fit(x_df)
        if self.encoder_type in ['dense_autoencoder', 'recurrent_autoencoder']:
            for i in x_df.columns:
                scaler = preprocessing.StandardScaler()
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
            x_train, x_val = model_selection.train_test_split(x_df, random_state=1)

            early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=0,
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

    @timeit
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
            with open(f'{self.file_path}.pkl', 'wb') as f:
                pickle.dump(self.model, f)

        if self.encoder_type == 'dense_autoencoder':
            models.save_model(self.encoder, f'{self.file_path}_encoder.pkl')
            models.save_model(self.autoencoder, f'{self.file_path}_autoencoder.pkl')
        with open(f'{self.scaler_file_path}.pkl'.format(file_path=self.scaler_file_path), 'wb') as f:
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
