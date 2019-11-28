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
                data_format):
    def f(input):
        for i in range(pooling_layers):
            for _ in range(1, conv_layers_per_pooling_layer):
                input = eval(convolution_type)(filters=convolutional_filters, kernel_size=convolutional_kernel_size,
                                               activation=activation, data_format=data_format)(input)
            if pooling_algorithm:
                input = eval(pooling_algorithm)(pool_size=convolutional_pool_size)(input)
        return input

    return f


def recurrent_cell(recurrent_bi_direction, recurrent_layers, recurrent_type, recurrent_layers_width, return_sequences):
    def f(input):
        if recurrent_bi_direction:
            if recurrent_layers == 1:
                x_rnn = layers.Bidirectional(
                    eval(recurrent_type)(recurrent_layers_width, return_sequences=return_sequences))(input)
            elif recurrent_layers == 2:
                x_rnn = layers.Bidirectional(eval(recurrent_type)(recurrent_layers_width, return_sequences=True))(
                    input)
                x_rnn = layers.Bidirectional(
                    eval(recurrent_type)(recurrent_layers_width, return_sequences=return_sequences))(x_rnn)
            else:
                raise NotImplemented
        else:
            if recurrent_layers == 1:
                x_rnn = eval(recurrent_type)(recurrent_layers_width)(input)
            elif recurrent_layers == 2:
                x_rnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=True)(input)
                x_rnn = eval(recurrent_type)(recurrent_layers_width, return_sequences=return_sequences)(x_rnn)
            else:
                raise NotImplemented
        return x_rnn

    return f


def recurrent_convolutional_cell(convolutional_filters,
                                 convolutional_kernel_size,
                                 activation,
                                 pooling_algorithm,
                                 convolutional_pool_size,
                                 convolution_type,
                                 conv_layers_per_pooling_layer,
                                 data_format,
                                 recurrent_bi_direction,
                                 recurrent_convolutional_layers, recurrent_type, recurrent_layers_width):
    def f(input):
        for i in range(recurrent_convolutional_layers):
            input = recurrent_cell(recurrent_bi_direction, 1, recurrent_type, recurrent_layers_width, True)(input)
            input = conv1d_cell(convolutional_filters, convolutional_kernel_size, activation,
                                pooling_algorithm, convolutional_pool_size, convolution_type, 1,
                                conv_layers_per_pooling_layer, data_format)(input)
        return input

    return f


def convolutional_recurrent_cell(convolutional_filters,
                                 convolutional_kernel_size,
                                 activation,
                                 pooling_algorithm,
                                 convolutional_pool_size,
                                 convolution_type,
                                 conv_layers_per_pooling_layer,
                                 data_format,
                                 recurrent_bi_direction,
                                 convolutional_recurrent_layers, recurrent_type, recurrent_layers_width,
                                 return_sequences):
    def f(input):
        for i in range(convolutional_recurrent_layers):
            input = conv1d_cell(convolutional_filters, convolutional_kernel_size, activation,
                                pooling_algorithm, convolutional_pool_size, convolution_type, 1,
                                conv_layers_per_pooling_layer, data_format)(input)
            if i == convolutional_recurrent_layers - 1:
                input = recurrent_cell(recurrent_bi_direction, 1, recurrent_type, recurrent_layers_width,
                                       return_sequences=return_sequences)(input)
            else:
                input = recurrent_cell(recurrent_bi_direction, 1, recurrent_type, recurrent_layers_width,
                                       return_sequences=True)(input)
        return input

    return f


def full_layer(convolutional_filters,
               convolutional_kernel_size,
               activation,
               pooling_algorithm,
               convolutional_pool_size,
               convolution_type,
               conv_layers_per_pooling_layer,
               data_format,
               recurrent_bi_direction,
               recurrent_type, recurrent_layers_width):
    def f(input):
        x_cnn1 = conv1d_cell(convolutional_filters, convolutional_kernel_size, activation, pooling_algorithm,
                             convolutional_pool_size, convolution_type, 1,
                             conv_layers_per_pooling_layer, data_format='channels_last')(input)
        x_cnn2 = conv1d_cell(convolutional_filters, convolutional_kernel_size, activation, pooling_algorithm,
                             convolutional_pool_size, convolution_type, 1,
                             conv_layers_per_pooling_layer, data_format='channels_first')(input)
        x_rnn = recurrent_cell(recurrent_bi_direction, 1, recurrent_type, recurrent_layers_width, True)(
            input)
        x_rnn_cnn = recurrent_convolutional_cell(convolutional_filters, convolutional_kernel_size,
                                                 activation, pooling_algorithm, convolutional_pool_size,
                                                 convolution_type,
                                                 conv_layers_per_pooling_layer, data_format, recurrent_bi_direction,
                                                 1, recurrent_type, recurrent_layers_width)(input)
        x_cnn_rnn = convolutional_recurrent_cell(convolutional_filters, convolutional_kernel_size,
                                                 activation, pooling_algorithm, convolutional_pool_size,
                                                 convolution_type,
                                                 conv_layers_per_pooling_layer, data_format, recurrent_bi_direction,
                                                 1, recurrent_type, recurrent_layers_width, return_sequences=True)(input)
        x = layers.Concatenate(name='x_cnn1_x_cnn2')([x_cnn1, x_cnn2])
        x = layers.Concatenate(name='x_cnn1_x_cnn2_x_rnn')([x, x_rnn])
        x = layers.Concatenate(name='x_cnn1_x_cnn2_x_rnn_x_rnn_cnn')([x, x_rnn_cnn])
        x = layers.Concatenate(name='x_cnn1_x_cnn2_x_rnn_x_rnn_cnn_x_cnn_rnn')([x, x_cnn_rnn])

        x = layers.Concatenate(name='full_concatenate')([x_cnn1, x_cnn2, x_rnn, x_rnn_cnn, x_cnn_rnn])
        return x

    return f


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
                 use_dense_cell,
                 dense_layers,
                 use_x1,
                 use_x2,
                 optimizer_algorithm,
                 activation,
                 conv_layers_per_pooling_layer,
                 use_full_cell,
                 full_cell_layers):

    if use_x1:
        input_layer = layers.Input(shape=input_shape1)

        if use_resnet:
            x = layers.Flatten()(input_layer)

        if use_full_cell:
            x_full = full_layer(convolutional_filters=convolutional_filters,
               convolutional_kernel_size=convolutional_kernel_size,
               activation=activation,
               pooling_algorithm=pooling_algorithm,
               convolutional_pool_size=convolutional_pool_size,
               convolution_type=convolution_type,
               conv_layers_per_pooling_layer=conv_layers_per_pooling_layer,
               data_format='channels_last',
               recurrent_bi_direction=recurrent_bi_direction,
               recurrent_type=recurrent_type,
               recurrent_layers_width=recurrent_layers_width)(input_layer)

            x_full = layers.Flatten(name = 'use_full_cell_flatten')(x_full)

            if not use_resnet:
                x = x_full
            else:
                x = layers.Concatenate()([x, x_full])

        if use_convolution_cell:
            x_conv = conv1d_cell(convolutional_filters, convolutional_kernel_size, activation, pooling_algorithm,
                                 convolutional_pool_size, convolution_type, convolutional_layers,
                                 conv_layers_per_pooling_layer, data_format='channels_last')(input_layer)
            x_conv = layers.Flatten(name='convolution_cell')(x_conv)
            if not use_resnet and not use_full_cell:
                x = x_conv
            else:
                x = layers.Concatenate()([x, x_conv])

        if use_recurrent_cell:
            x_rnn = recurrent_cell(recurrent_bi_direction=recurrent_bi_direction, recurrent_layers=recurrent_layers,
                                   recurrent_type=recurrent_type, recurrent_layers_width=recurrent_layers_width,
                                   return_sequences=False)(
                input_layer)
            if not (use_resnet or use_convolution_cell or use_full_cell):
                x = x_rnn
            else:
                x = layers.Concatenate()([x, x_rnn])

        if use_recurrent_convolutional_cell:
            x_rnn_cnn = recurrent_convolutional_cell(convolutional_filters=convolutional_filters,
                                                     convolutional_kernel_size=convolutional_kernel_size,
                                                     activation=activation,
                                                     pooling_algorithm=pooling_algorithm,
                                                     convolutional_pool_size=convolutional_pool_size,
                                                     convolution_type=convolution_type,
                                                     conv_layers_per_pooling_layer=conv_layers_per_pooling_layer,
                                                     data_format='channels_last',
                                                     recurrent_bi_direction=recurrent_bi_direction,
                                                     recurrent_convolutional_layers=recurrent_convolutional_layers,
                                                     recurrent_type=recurrent_type,
                                                     recurrent_layers_width=recurrent_layers_width)(input_layer)
            x_rnn_cnn = layers.Flatten(name='recurrent_convolutional_cell_flatten')(x_rnn_cnn)

            if not (use_resnet or use_convolution_cell or use_recurrent_cell or use_full_cell):
                x = x_rnn_cnn
            else:
                x = layers.Concatenate()([x, x_rnn_cnn])

        if use_convolutional_recurrent_cell:
            x_cnn_rnn = convolutional_recurrent_cell(convolutional_filters=convolutional_filters,
                                                     convolutional_kernel_size=convolutional_kernel_size,
                                                     activation=activation,
                                                     pooling_algorithm=pooling_algorithm,
                                                     convolutional_pool_size=convolutional_pool_size,
                                                     convolution_type=convolution_type,
                                                     conv_layers_per_pooling_layer=conv_layers_per_pooling_layer,
                                                     data_format='channels_last',
                                                     recurrent_bi_direction=recurrent_bi_direction,
                                                     convolutional_recurrent_layers=convolutional_recurrent_layers,
                                                     recurrent_type=recurrent_type,
                                                     recurrent_layers_width=recurrent_layers_width,
                                                     return_sequences=False)(input_layer)

            if not (use_resnet or use_convolution_cell or use_recurrent_cell or use_recurrent_convolutional_cell or use_full_cell):
                x = x_cnn_rnn
            else:
                x = layers.Concatenate()([x, x_cnn_rnn])

        if use_dense_cell:
            x_dnn = layers.Flatten(name='flatten15')(input_layer)
            for i in range(dense_layers):
                x_dnn = layers.Dense(dense_layers_width)(x_dnn)

            if not (
                    use_resnet or use_convolution_cell or use_recurrent_cell or use_recurrent_convolutional_cell or use_convolutional_recurrent_cell or use_full_cell):
                x = x_dnn
            else:
                x = layers.Concatenate()([x, x_dnn])

    if use_x2:
        input_layer2 = layers.Input(shape=input_shape2)
        if use_x1:
            x = layers.Concatenate()([x, input_layer2])
        else:
            x = input_layer2

    for i in range(dense_top_layers):
        x = layers.Dense(dense_layers_width)(x)

    out = layers.Dense(2, activation='softmax')(x)

    if use_x1 and use_x2:
        model = models.Model(inputs=[input_layer, input_layer2], outputs=out)
    elif use_x1:
        model = models.Model(inputs=input_layer, outputs=out)
    else:
        model = models.Model(inputs=input_layer2, outputs=out)

    model.compile(optimizer=eval(optimizer_algorithm),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
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
    use_convolution_cell = [True, False]
    convolutional_layers = [1, 2]
    conv_layers_per_pooling_layer = [1, 2]

    recurrent_type = ['layers.RNN', 'layers.LSTM', 'layers.GRU']
    recurrent_bi_direction = [True, False]
    use_recurrent_cell = [True, False]
    recurrent_layers = [1, 2]

    use_recurrent_convolutional_cell = [True, False]
    recurrent_convolutional_layers = [1, 2]

    use_convolutional_recurrent_cell = [True, False]

    convolutional_recurrent_layers = [1, 2]

    use_full_cell = [False]
    full_cell_layers = [1]

    use_dense_cell = [True, False]
    dense_layers = [1, 2]

    use_x2 = [True, False]
    use_x1 = [True, False]

    activation = ['relu', 'elu', 'sigmoid']

    # history_lengths = [32]
    transpose_history_data = [True, False]
    history_lengths = [16]
    # transpose_history_data = [True, False]
    # transformers = ['QuantileTransformer', 'StandardScaler']

    optimizer_algorithm = ['optimizers.SGD()',
                           'optimizers.RMSprop()',
                           'optimizers.Adagrad()',
                           'optimizers.Adadelta()',
                           'optimizers.Adamax()',
                           'optimizers.Nadam()']

    dm = DataManager(fill_nans=True, data_scaling='QuantileTransformer', testing=1000)
    dm.update_raw_datasets()
    del dm

    dms = dict()
    for i in history_lengths:
        for j in transpose_history_data:
            dm = DataManager(fill_nans=True, data_scaling='QuantileTransformer')
            dm.build_timeseries(history_length=i, transpose_history_data=j)
            x1, x2, y, columns = dm.get_labeled_data(history_length=i, transpose_history_data=j, get_history_data=True)
            print('get_labeled_data output', i, j, x1.shape, x2.shape, y.shape, len(columns))
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
        choice_dict = {'convolutional_filters': random.choice(convolutional_filters),
                       'convolutional_kernel_size': random.choice(convolutional_kernel_size),
                       'convolutional_pool_size': random.choice(convolutional_pool_size),
                       'dense_top_layers': random.choice(dense_top_layers),
                       'dense_layers_width': random.choice(dense_layers_width),
                       'recurrent_layers_width': random.choice(recurrent_layers_width),
                       'pooling_algorithm': random.choice(pooling_algorithm),
                       'use_resnet': random.choice(use_resnet),
                       'convolution_type': random.choice(convolution_type),
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
                       'use_dense_cell': random.choice(use_dense_cell),
                       'dense_layers': random.choice(dense_layers),
                       'use_x1': random.choice(use_x1),
                       'use_x2': random.choice(use_x2),
                       'optimizer_algorithm': random.choice(optimizer_algorithm),
                       'activation': random.choice(activation),
                       'conv_layers_per_pooling_layer': random.choice(conv_layers_per_pooling_layer),
                       'use_full_cell':random.choice(use_full_cell),
                       'full_cell_layers':random.choice(full_cell_layers)}


        try:
            assert choice_dict['use_x1'] or choice_dict['use_x2']

            history_length_choice = random.choice(history_lengths)
            transpose_history_data_choice = random.choice(transpose_history_data)
            key = (history_length_choice, transpose_history_data_choice)
            choice_dict['input_shape1'] = dms[key]['x1_train'].shape[1:]
            choice_dict['input_shape2'] = dms[key]['x2_train'].shape[1:]

            print(dms[key]['x1_train'].shape, dms[key]['x2_train'].shape)
            model = get_nn_model(**choice_dict)

            choice_dict['num_of_params'] = model.count_params()
            choice_dict['history_lengths'] = history_length_choice
            choice_dict['transpose_history_data'] = transpose_history_data_choice
            print(choice_dict)
            start_time = time.time()
            cb = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=2,
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
            elif choice_dict['use_x1']:
                model.fit(dms[key]['x1_train'],
                          dms[key]['y_train'],
                          validation_data=(dms[key]['x1_val'],
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
            model = models.load_model('{}/test.h5'.format(data_path))

            if choice_dict['use_x1'] and choice_dict['use_x2']:
                preds = model.predict([dms[key]['x1_test'], dms[key]['x2_test']])
            elif choice_dict['use_x1']:
                preds = model.predict(dms[key]['x1_test'])
            elif choice_dict['use_x2']:
                preds = model.predict(dms[key]['x2_test'])
            else:
                raise Exception('invalid setup')

            preds = np.rint(preds[:, 1]).astype(int)
            truth = dms[key]['y_test'][:, 1]

            print(preds[:5], truth[:5])
            score = accuracy_score(
                truth,
                preds)
            choice_dict['accuracy'] = score

            del model, preds, score, cb, mcp_save
            gc.collect()

        except AssertionError:
            continue
        except:
            traceback.print_exc()

        results.append(choice_dict)
        print(results)
        pd.DataFrame.from_dict(results).to_csv(f'{data_path}/nn_architectures.csv', index=False)


if __name__ == '__main__':
    optimize_nn()
