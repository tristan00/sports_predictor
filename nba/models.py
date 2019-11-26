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

def run_naive_model():
    dm = DataManager()
    x, y = dm.get_labeled_data()
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x_train, x_val, y_train, y_val = train_test_split(x, y)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    print(accuracy_score(y_val, rf.predict(x_val)))


def get_nn_model(input_shape1, input_shape2, filters, kernel_size, pool_size, dense_top_layers, dense_layers_width,
                  convolutional_layers, recurrent_layers, dnn_layers, network_type):
    input_layer = layers.Input(shape=input_shape1)


    if network_type == 'cnn':
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_layer)
        x = layers.MaxPooling1D(pool_size=pool_size)(x)

        for i in range(convolutional_layers):
            x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=pool_size)(x)

        x = layers.Flatten()(x)

    elif network_type == 'rnn':
        if recurrent_layers > 1:
            x = layers.GRU(dense_layers_width, return_sequences=True)(input_layer)
        else:
            x = layers.GRU(dense_layers_width)(input_layer)

        for i in range(1, convolutional_layers):
            if recurrent_layers > i + 1:
                x = layers.GRU(dense_layers_width, return_sequences=True)(input_layer)
            else:
                x = layers.GRU(dense_layers_width)(input_layer)

        # x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(input_layer)
    if network_type == 'dnn':
        for i in range(dnn_layers):
            x = layers.Dense(dense_layers_width)(x)

    input_layer2 = layers.Input(shape=input_shape2)
    x = layers.Concatenate()([x, input_layer2])

    for i in range(dense_top_layers):
        x = layers.Dense(dense_layers_width)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[input_layer, input_layer2], outputs=out)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def optimize_nn():
    filters = list(range(1,128))
    kernel_size = list(range(1,10))
    pool_size = list(range(1,10))
    dense_top_layers = [1, 2]
    dense_layers_width = [64]
    convolutional_layers = [1, 2, 3]
    rnn_layers = [1, 2, 3]
    dnn_layers = [1]
    network_type = ['cnn', 'rnn']
    # network_type = ['cnn']

    history_lengths = [4, 8, 16, 32, 64, 128]
    # history_lengths = [64]
    transpose_history_data = [True, False]

    dm = DataManager()
    # dm.update_raw_datasets()

    dms = dict()
    for i in history_lengths:
        for j in transpose_history_data:
            # dm.build_timeseries(i, j)
            # dm.combine_timeseries(i, j)
            x1, x2, y = dm.get_labeled_data(i, j)
            x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1, x2, y, random_state=1)
            x1_val, x1_test, x2_val, x2_test, y_val, y_test = train_test_split(x1_val, x2_val, y_val, train_size=.5, random_state=1)
            dms[(i, j)] = {'x1_train': x1_train,
                           'x1_val': x1_val,
                           'x1_test': x1_test,
                           'x2_train': x2_train,
                           'x2_val': x2_val,
                           'x2_test': x2_test,
                           'y_train': y_train,
                           'y_val': y_val,
                           'y_test': y_test}

    results = list()
    for i in range(10000):
        for network_type_choice in network_type:
            model_built = False
            while not model_built:
                try:
                    filters_choice = random.choice(filters)
                    kernel_size_choice = random.choice(kernel_size)
                    pool_size_choice = random.choice(pool_size)
                    dense_top_layers_choice = random.choice(dense_top_layers)
                    dense_layers_width_choice = random.choice(dense_layers_width)
                    convolutional_layers_choice = random.choice(convolutional_layers)
                    history_lengths_choice = random.choice(history_lengths)
                    transpose_history_choice = random.choice(transpose_history_data)
                    recurrent_layers_choice = random.choice(rnn_layers)
                    dnn_layers_choice = random.choice(dnn_layers)

                    key = (history_lengths_choice, transpose_history_choice)

                    print({'filters': filters_choice,
                                    'kernel_size': kernel_size_choice,
                                    'pool_size': pool_size_choice,
                                    'dense_top_layers': dense_top_layers_choice,
                                    'dense_layers_width': dense_layers_width_choice,
                                    'convolutional_layers': convolutional_layers_choice,
                                    'recurrent_layers': recurrent_layers_choice,
                                    'dnn_layers': dnn_layers_choice,
                                    'network_type': network_type_choice,
                                    'history_lengths': history_lengths_choice,
                                    'transpose_history': transpose_history_choice
                                    })

                    model = get_nn_model(
                        input_shape1=(dms[key]['x1_train'].shape[1],
                                     dms[key]['x1_train'].shape[2]),
                        input_shape2=(dms[key]['x2_train'].shape[1],),
                        filters=filters_choice,
                        kernel_size=kernel_size_choice,
                        pool_size=pool_size_choice,
                        dense_top_layers=dense_top_layers_choice,
                        dense_layers_width=dense_layers_width_choice,
                        convolutional_layers=convolutional_layers_choice,
                        recurrent_layers=recurrent_layers_choice,
                        dnn_layers=dnn_layers_choice,
                        network_type=network_type_choice
                        )
                    cb = callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=1,
                                                 verbose=0, mode='auto')
                    mcp_save = callbacks.ModelCheckpoint('{}/test.h5'.format(data_path), save_best_only=True,
                                                         monitor='val_loss',
                                                         verbose=1)
                    model.fit([dms[key]['x1_train'], dms[key]['x2_train']],
                              dms[key]['y_train'],
                              validation_data=([dms[key]['x1_val'], dms[key]['x2_val']],
                                               dms[key]['y_val']),
                              callbacks=[cb, mcp_save], epochs=200, batch_size=128)

                    model = models.load_model('{}/test.h5'.format(data_path))
                    preds = np.rint(model.predict([dms[key]['x1_test'], dms[key]['x2_test']]))
                    score = accuracy_score(dms[(history_lengths_choice, transpose_history_choice)]['y_test'], preds.astype(int))

                    results.append({'filters': filters_choice,
                                    'kernel_size': kernel_size_choice,
                                    'pool_size': pool_size_choice,
                                    'dense_top_layers': dense_top_layers_choice,
                                    'dense_layers_width': dense_layers_width_choice,
                                    'convolutional_layers': convolutional_layers_choice,
                                    'recurrent_layers': recurrent_layers_choice,
                                    'dnn_layers': dnn_layers_choice,
                                    'network_type': network_type_choice,
                                    'history_lengths': history_lengths_choice,
                                    'transpose_history': transpose_history_choice,
                                    'accuracy': score
                                    })

                    results = sorted(results, key=lambda x: x['accuracy'], reverse=False)
                    print(results)

                    pd.DataFrame.from_dict(results).to_csv(f'{data_path}/nn_architectures.csv', index = False)
                    model_built = True
                except:
                    traceback.print_exc()


if __name__ == '__main__':
    optimize_nn()
