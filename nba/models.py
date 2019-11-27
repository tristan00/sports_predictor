from nba.process_data import DataManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks
from nba.common import data_path
import numpy as np
import random
import traceback
import pandas as pd
import gc


def run_naive_model():
    dm = DataManager()
    x, y = dm.get_labeled_data()
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x_train, x_val, y_train, y_val = train_test_split(x, y)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print(accuracy_score(y_val, rf.predict(x_val)))


def get_nn_model(input_shape1,
                 input_shape2,
                 convolutional_filters,
                 convolutional_kernel_size,
                 convolutional_pool_size,
                 dense_top_layers,
                 dense_layers_width,
                 recurrent_layers_width,
                 pooling_algorithm,
                 use_resnet,
                 convolution_type,
                 convolution_2_direction,
                 use_convolution_cell,
                 convolutional_layers,
                 recurrent_type,
                 recurrent_bi_direction,
                 use_recurrent_cell,
                 recurrent_layers,
                 use_recurrent_convolutional_cell,
                 recurrent_convolutional_layers,
                 use_convolutional_recurrent_cell,
                 convolutional_recurrent_layers,
                 use_convolutional_x_convolutional_cell,
                 convolutional_x_convolutional_layers,
                 use_dense_cell,
                 dense_layers,
                 use_x1,
                 use_x2):

    if use_x1:
        input_layer = layers.Input(shape=input_shape1)

        if use_resnet:
            x = layers.Flatten()(input_layer)

        if use_convolution_cell:
            if convolution_2_direction:
                x_cnn_1 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                 activation='relu', data_format='channels_last')(input_layer)
                x_cnn_2 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                 activation='relu', data_format='channels_first')(input_layer)
                if pooling_algorithm:
                    x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_1)
                    x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_2)
                for i in range(convolutional_layers):
                    x_cnn_1 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                     activation='relu', data_format='channels_last')(x_cnn_1)
                    x_cnn_2 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                     activation='relu', data_format='channels_first')(x_cnn_2)
                    if pooling_algorithm:
                        x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_1)
                        x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_2)
                x_cnn_1 = layers.Flatten()(x_cnn_1)
                x_cnn_2 = layers.Flatten()(x_cnn_2)
                x_conv = layers.Concatenate()([x_cnn_1, x_cnn_2])
            else:
                x_conv = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                activation='relu')(input_layer)
                if pooling_algorithm:
                    x_conv = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_conv)

                for i in range(convolutional_layers):
                    x_conv = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                    activation='relu')(x_conv)
                    if pooling_algorithm:
                        x_conv = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_conv)
                x_conv = layers.Flatten()(x_conv)
            if not use_resnet:
                x = x_conv
            else:
                x = layers.Concatenate()([x, x_conv])

        if use_recurrent_cell:
            if recurrent_bi_direction:
                if recurrent_layers == 1:
                    x_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(input_layer)
                elif recurrent_layers == 2:
                    x_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(
                        input_layer)
                    x_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_rnn)
                else:
                    raise NotImplemented
            else:
                if recurrent_layers == 1:
                    x_rnn = eval(recurrent_type)(recurrent_layers_width)(input_layer)
                elif recurrent_layers == 2:
                    x_rnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input_layer)
                    x_rnn = eval(recurrent_type)(recurrent_layers_width)(x_rnn)
                else:
                    raise NotImplemented
            if not (use_resnet or use_convolution_cell):
                x = x_rnn
            else:
                x = layers.Concatenate()([x, x_rnn])

        if use_recurrent_convolutional_cell:
            if recurrent_convolutional_layers == 1:
                if convolution_2_direction:
                    if recurrent_bi_direction:
                        x_rnn_cnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(input_layer)
                    else:
                        x_rnn_cnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input_layer)
                    x_rnn_cnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_last')(x_rnn_cnn)
                    x_rnn_cnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_first')(x_rnn_cnn)
                    x_rnn_cnn_1 = layers.Flatten()(x_rnn_cnn_1)
                    x_rnn_cnn_2 = layers.Flatten()(x_rnn_cnn_2)
                    x_rnn_cnn = layers.Concatenate()([x_rnn_cnn_2, x_rnn_cnn_1])
                else:
                    if recurrent_bi_direction:
                        x_rnn_cnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(input_layer)
                    else:
                        x_rnn_cnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input_layer)
                    x_rnn_cnn = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(x_rnn_cnn)
                    x_rnn_cnn = layers.Flatten()(x_rnn_cnn)
            elif recurrent_convolutional_layers == 2:
                if convolution_2_direction:
                    if recurrent_bi_direction:
                        x_rnn_cnn = layers.Bidirectional(
                            eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(input_layer)
                    else:
                        x_rnn_cnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input_layer)
                    x_rnn_cnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_last')(x_rnn_cnn)
                    x_rnn_cnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_first')(x_rnn_cnn)

                    for i in range(convolutional_layers):
                        if recurrent_bi_direction:
                            x_rnn_cnn_1 = layers.Bidirectional(
                                eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(x_rnn_cnn_1)
                            x_rnn_cnn_2 = layers.Bidirectional(
                                eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(x_rnn_cnn_2)
                        else:
                            x_rnn_cnn_1 = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(x_rnn_cnn_1)
                            x_rnn_cnn_2 = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(x_rnn_cnn_2)

                        x_rnn_cnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                             kernel_size=convolutional_kernel_size, activation='relu',
                                                             data_format='channels_last')(x_rnn_cnn_1)
                        x_rnn_cnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                             kernel_size=convolutional_kernel_size, activation='relu',
                                                             data_format='channels_first')(x_rnn_cnn_2)
                        if pooling_algorithm:
                            x_rnn_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_rnn_cnn_1)
                            x_rnn_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_rnn_cnn_2)
                    x_rnn_cnn_1 = layers.Flatten()(x_rnn_cnn_1)
                    x_rnn_cnn_2 = layers.Flatten()(x_rnn_cnn_2)
                    x_rnn_cnn = layers.Concatenate()([x_rnn_cnn_1, x_rnn_cnn_2])
                else:
                    if recurrent_bi_direction:
                        x_rnn_cnn = layers.Bidirectional(
                            eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(input_layer)
                    else:
                        x_rnn_cnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input_layer)
                    x_rnn_cnn = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(x_rnn_cnn)
                    x_rnn_cnn = layers.Flatten()(x_rnn_cnn)

                    for i in range(convolutional_layers):
                        if recurrent_bi_direction:
                            x_rnn_cnn = layers.Bidirectional(
                                eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(x_rnn_cnn)
                        else:
                            x_rnn_cnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(x_rnn_cnn)

                        x_rnn_cnn = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_last')(x_rnn_cnn)
                        if pooling_algorithm:
                            x_rnn_cnn = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_rnn_cnn)
            else:
                raise NotImplemented

            if not (use_resnet or use_convolution_cell or use_recurrent_cell):
                x = x_rnn_cnn
            else:
                x = layers.Concatenate()([x, x_rnn_cnn])

        if use_convolutional_recurrent_cell:
            if convolutional_recurrent_layers == 1:
                if convolution_2_direction:
                    x_cnn_rnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_last')(input_layer)
                    x_cnn_rnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_first')(input_layer)

                    if recurrent_bi_direction:
                        x_cnn_rnn_1 = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn_1)
                        x_cnn_rnn_2 = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn_2)
                    else:
                        x_cnn_rnn_1 = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn_1)
                        x_cnn_rnn_2 = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn_2)

                    x_cnn_rnn_1 = layers.Flatten()(x_cnn_rnn_1)
                    x_cnn_rnn_2 = layers.Flatten()(x_cnn_rnn_2)
                    x_cnn_rnn = layers.Concatenate()([x_cnn_rnn_1, x_cnn_rnn_2])


                else:
                    x_cnn_rnn = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(input_layer)
                    if recurrent_bi_direction:
                        x_cnn_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn)
                    else:
                        x_cnn_rnn = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn)
                    x_cnn_rnn = layers.Flatten()(x_cnn_rnn)

            elif convolutional_recurrent_layers == 2:
                if convolution_2_direction:

                    x_cnn_rnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_last')(input_layer)
                    x_cnn_rnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu',
                                                         data_format='channels_first')(input_layer)

                    if recurrent_bi_direction:
                        x_cnn_rnn_1 = layers.Bidirectional(
                            eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(x_cnn_rnn_1)
                        x_cnn_rnn_2 = layers.Bidirectional(
                            eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(x_cnn_rnn_2)
                    else:
                        x_cnn_rnn_1 = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(x_cnn_rnn_1)
                        x_cnn_rnn_2 = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(x_cnn_rnn_2)

                    for i in range(convolutional_layers):
                        x_cnn_rnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                             kernel_size=convolutional_kernel_size, activation='relu',
                                                             data_format='channels_last')(x_cnn_rnn_1)
                        x_cnn_rnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                             kernel_size=convolutional_kernel_size, activation='relu',
                                                             data_format='channels_first')(x_cnn_rnn_2)

                        if pooling_algorithm:
                            x_cnn_rnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_rnn_1)
                            x_cnn_rnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_rnn_2)

                        if recurrent_bi_direction:
                            x_cnn_rnn_1 = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn_1)
                            x_cnn_rnn_2 = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn_2)
                        else:
                            x_cnn_rnn_1 = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn_1)
                            x_cnn_rnn_2 = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn_2)

                    x_cnn_rnn = layers.Concatenate()([x_cnn_rnn_1, x_cnn_rnn_2])
                else:
                    if recurrent_bi_direction:
                        x_cnn_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(input_layer)
                    else:
                        x_cnn_rnn = eval(recurrent_type)(recurrent_layers_width)(input_layer)
                    x_cnn_rnn = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(x_cnn_rnn)
                    x_cnn_rnn = layers.Flatten()(x_cnn_rnn)

                    for i in range(convolutional_layers):
                        if recurrent_bi_direction:
                            x_cnn_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width))(x_cnn_rnn)
                        else:
                            x_cnn_rnn = eval(recurrent_type)(recurrent_layers_width)(x_cnn_rnn)

                        x_cnn_rnn = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_last')(x_cnn_rnn)
                        if pooling_algorithm:
                            x_cnn_rnn = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_rnn)
            else:
                raise NotImplemented

            if not (use_resnet or use_convolution_cell or use_recurrent_cell or use_recurrent_convolutional_cell):
                x = x_cnn_rnn
            else:
                x = layers.Concatenate()([x, x_cnn_rnn])

        if use_convolutional_x_convolutional_cell:
            if convolution_2_direction:
                x_cnn_x_cnn_1 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(input_layer)
                x_cnn_x_cnn_2 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_first')(input_layer)

                if pooling_algorithm:
                    x_cnn_x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_1)
                    x_cnn_x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_2)

                x_cnn_x_cnn_1 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_first')(x_cnn_x_cnn_1)
                x_cnn_x_cnn_2 = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                       activation='relu', data_format='channels_last')(x_cnn_x_cnn_2)

                if pooling_algorithm:
                    x_cnn_x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_1)
                    x_cnn_x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_2)

                for i in range(convolutional_x_convolutional_layers):
                    x_cnn_x_cnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_last')(x_cnn_x_cnn_1)
                    x_cnn_x_cnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_first')(x_cnn_x_cnn_2)

                    if pooling_algorithm:
                        x_cnn_x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_1)
                        x_cnn_x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_2)

                    x_cnn_x_cnn_1 = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_first')(x_cnn_x_cnn_1)
                    x_cnn_x_cnn_2 = eval(convolution_type)(filters=convolutional_filters,
                                                           kernel_size=convolutional_kernel_size, activation='relu',
                                                           data_format='channels_last')(x_cnn_x_cnn_2)

                    if pooling_algorithm:
                        x_cnn_x_cnn_1 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_1)
                        x_cnn_x_cnn_2 = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn_2)

                x_cnn_x_cnn_1 = layers.Flatten()(x_cnn_x_cnn_1)
                x_cnn_x_cnn_2 = layers.Flatten()(x_cnn_x_cnn_2)
                x_cnn_x_cnn = layers.Concatenate()([x_cnn_x_cnn_1, x_cnn_x_cnn_2])
            else:
                x_cnn_x_cnn = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                                     activation='relu')(input_layer)
                if pooling_algorithm:
                    x_cnn_x_cnn = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn)

                for i in range(convolutional_layers):
                    x_cnn_x_cnn = eval(convolution_type)(filters=convolutional_filters,
                                                         kernel_size=convolutional_kernel_size, activation='relu')(
                        x_cnn_x_cnn)
                    if pooling_algorithm:
                        x_cnn_x_cnn = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(x_cnn_x_cnn)
                x_cnn_x_cnn = layers.Flatten()(x_cnn_x_cnn)

            if not use_resnet or use_convolution_cell or use_recurrent_cell or use_recurrent_convolutional_cell or use_convolutional_recurrent_cell:
                x = x_cnn_x_cnn
            else:
                x = layers.Concatenate()([x, x_cnn_x_cnn])

        if use_dense_cell:
            x_dnn = layers.Flatten()(input_layer)
            for i in range(dense_layers):
                x_dnn = layers.Dense(dense_layers_width)(x_dnn)

            if not (
                    use_resnet or use_convolution_cell or use_recurrent_cell or use_recurrent_convolutional_cell or use_convolutional_recurrent_cell or use_convolutional_x_convolutional_cell):
                x = x_dnn
            else:
                x = layers.Concatenate()([x, x_dnn])

    if use_x2 and not use_x1:
        x = layers.Input(shape=input_shape2)
    if use_x1 and use_x2:
        input_layer2 = layers.Input(shape=input_shape2)
        x = layers.Concatenate()([x, input_layer2])

    for i in range(dense_top_layers):
        x = layers.Dense(dense_layers_width)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    if use_x1 and use_x2:
        model = models.Model(inputs=[input_layer, input_layer2], outputs=out)
    elif use_x1:
        model = models.Model(inputs=[input_layer2], outputs=out)
    elif use_x2:
        model = models.Model(inputs=[input_layer2], outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def optimize_nn():
    convolutional_filters = list(range(16, 128))
    convolutional_kernel_size = list(range(1, 10))
    convolutional_pool_size = list(range(1, 5))
    dense_top_layers = [0, 1, 2, 3]
    dense_layers_width = list(range(16, 128))
    recurrent_layers_width = list(range(16, 128))

    pooling_algorithm = ['layers.MaxPooling1D', 'layers.AveragePooling1D', None]
    use_resnet = [True, False]

    convolution_type = ['layers.Conv1D', 'layers.SeparableConv1D', 'layers.LocallyConnected1D']
    convolution_2_direction = [True, False]
    use_convolution_cell = [True, False]
    convolutional_layers = [1, 2]

    recurrent_type = ['layers.RNN', 'layers.LSTM', 'layers.GRU']
    recurrent_bi_direction = [True, False]
    use_recurrent_cell = [True, False]
    recurrent_layers = [1, 2]

    use_recurrent_convolutional_cell = [True, False]
    recurrent_convolutional_layers = [1, 2]

    use_convolutional_recurrent_cell = [True, False]
    convolutional_recurrent_layers = [1, 2]

    use_convolutional_x_convolutional_cell = [True, False]
    convolutional_x_convolutional_layers = [1, 2]

    use_dense_cell = [True, False]
    dense_layers = [1, 2]

    use_x2 = [True, False]
    use_x1 = [True, False]

    # history_lengths = [16]
    # transpose_history_data = [False]
    history_lengths = [16, 32, 64]
    transpose_history_data = [True, False]

    dm = DataManager()
    dm.update_raw_datasets()
    del dm

    dms = dict()
    for i in history_lengths:
        for j in transpose_history_data:
            dm = DataManager()
            dm.build_timeseries(i, j)
            dm.combine_timeseries(i, j)
            x1, x2, y, _ = dm.get_labeled_data(history_length = i, transpose_history_data = j, get_history_data = True)
            print(x1.shape, x2.shape, y.shape)
            del dm
            x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1, x2, y, random_state=1)
            x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_val, x2_val, y_val, train_size=.5,
                                                                               random_state=1)
            dms[(i, j)] = {'x1_train': x1_train,
                           'x1_val': x1_val,
                           'x1_test': x1_test,
                           'x2_train': x2_train,
                           'x2_val': x2_val,
                           'x2_test': x2_test,
                           'y_train': y_train,
                           'y_val': y_val,
                           'y_test': y_test}

    try:
        results = pd.read_csv(f'{data_path}/nn_architectures.csv').to_dict(orient='records')
    except:
        results = list()
    while True:
        try:
            choice_dict = {'convolutional_filters': random.choice(convolutional_filters),
                           'convolutional_kernel_size': random.choice(convolutional_kernel_size),
                           'convolutional_pool_size': random.choice(convolutional_pool_size),
                           'dense_top_layers': random.choice(dense_top_layers),
                           'dense_layers_width': random.choice(dense_layers_width),
                           'recurrent_layers_width': random.choice(recurrent_layers_width),
                           'pooling_algorithm': random.choice(pooling_algorithm),
                           'use_resnet': random.choice(use_resnet),
                           'convolution_type': random.choice(convolution_type),
                           'convolution_2_direction': random.choice(convolution_2_direction),
                           'use_convolution_cell': random.choice(use_convolution_cell),
                           'convolutional_layers': random.choice(convolutional_layers),
                           'recurrent_type': random.choice(recurrent_type),
                           'recurrent_bi_direction': random.choice(recurrent_bi_direction),
                           'use_recurrent_cell': random.choice(use_recurrent_cell),
                           'recurrent_layers': random.choice(recurrent_layers),
                           'use_recurrent_convolutional_cell': random.choice(use_recurrent_convolutional_cell),
                           'recurrent_convolutional_layers': random.choice(recurrent_convolutional_layers),
                           'use_convolutional_recurrent_cell': random.choice(use_convolutional_recurrent_cell),
                           'convolutional_recurrent_layers': random.choice(convolutional_recurrent_layers),
                           'use_convolutional_x_convolutional_cell': random.choice(
                               use_convolutional_x_convolutional_cell),
                           'convolutional_x_convolutional_layers': random.choice(convolutional_x_convolutional_layers),
                           'use_dense_cell': random.choice(use_dense_cell),
                           'dense_layers': random.choice(dense_layers),
                           'use_x1': random.choice(use_x1),
                           'use_x2': random.choice(use_x2)}

            if not choice_dict['use_x1'] and not choice_dict['use_x2']:
                continue

            history_length_choice = random.choice(history_lengths)
            transpose_history_data_choice = random.choice(transpose_history_data)
            key = (history_length_choice, transpose_history_data_choice)
            choice_dict['input_shape1'] = dms[key]['x1_train'].shape[1:]
            choice_dict['input_shape2'] = dms[key]['x2_train'].shape[1:]

            print(dms[key]['x1_train'].shape, dms[key]['x2_train'].shape)
            model = get_nn_model(**choice_dict)

            choice_dict['history_lengths'] = random.choice(history_lengths),
            choice_dict['transpose_history_data'] = random.choice(transpose_history_data)
            print(choice_dict)

            cb = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=1,
                                         verbose=0, mode='auto')
            mcp_save = callbacks.ModelCheckpoint('{}/test.h5'.format(data_path), save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

            if choice_dict['use_x1'] and choice_dict['use_x2']:
                model.fit([dms[key]['x1_train'], dms[key]['x2_train']],
                      dms[key]['y_train'],
                      validation_data=([dms[key]['x1_val'], dms[key]['x2_val']],
                                       dms[key]['y_val']),
                      callbacks=[cb, mcp_save], epochs=200, batch_size=128)
            elif use_x1:
                model.fit(dms[key]['x1_train'],
                      dms[key]['y_train'],
                      validation_data=(dms[key]['x1_val'],
                                       dms[key]['y_val']),
                      callbacks=[cb, mcp_save], epochs=200, batch_size=128)
            elif use_x2:
                model.fit(dms[key]['x2_train'],
                      dms[key]['y_train'],
                      validation_data=(dms[key]['x2_val'],
                                       dms[key]['y_val']),
                      callbacks=[cb, mcp_save], epochs=200, batch_size=128)

            del model
            model = models.load_model('{}/test.h5'.format(data_path))

            if choice_dict['use_x1'] and choice_dict['use_x2']:
                preds = np.rint(model.predict([dms[key]['x1_test'], dms[key]['x2_test']]))
            elif use_x1:
                preds = np.rint(model.predict(dms[key]['x1_test']))
            elif use_x2:
                preds = np.rint(model.predict(dms[key]['x2_test']))

            score = accuracy_score(
                dms[key]['y_test'],
                preds.astype(int))
            choice_dict['accuracy'] = score

            results.append(choice_dict)

            del model, preds, score, cb, mcp_save
            gc.collect()

            results = sorted(results, key=lambda x: x['accuracy'], reverse=False)
            print(results)

            pd.DataFrame.from_dict(results).to_csv(f'{data_path}/nn_architectures.csv', index=False)
        except:
            traceback.print_exc()


if __name__ == '__main__':
    optimize_nn()
