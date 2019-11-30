import pandas as pd
import lightgbm
import pandas
from sklearn import metrics, model_selection
from nba.common import (
    timeit,
)
from nba.data_pipeline import load_general_feature_file


class Model():
    max_iter = 1000000

    lightgbm_max_iter = 10000
    lightgbm_early_stopping_rounds = 100

    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.transformers_dict = dict()

    @timeit
    def fit(self, x, y):
        print('entered fit, x shape: {}'.format(x.shape))
        self.transformers_dict = dict()
        self.columns = x.columns

        if self.model_type == 'lightgbm':
            x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y)
            lgtrain = lightgbm.Dataset(x_train, y_train)
            lgvalid = lightgbm.Dataset(x_val, y_val)

            self.model = lightgbm.train(
                self.model_params,
                lgtrain,
                num_boost_round=self.lightgbm_max_iter,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train', 'valid'],
                early_stopping_rounds=self.lightgbm_early_stopping_rounds,
                verbose_eval=100
            )

    @timeit
    def predict(self, x):
        if self.model_type == 'lightgbm':
            return self.model.predict(x, num_iteration=self.model.best_iteration)

    def get_metric(self, x, y):
        preds = self.predict(x)
        return metrics.accuracy_score(y, preds)

    def evaluate(self):
        if self.model_type == 'lightgbm':
            output = []

            for i, j in zip(self.columns, self.model.feature_importance('gain', iteration=self.model.best_iteration)):
                output.append({'column': i, 'feature_importance': j})
            return pd.DataFrame.from_dict(output).sort_values('feature_importance', ascending=False)


def run_lightgbm_model(x, y):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_error',
        "learning_rate": 0.01,
        "max_depth": -1,
        'num_leaves': 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        'bagging_freq': 1,
    }
    model = Model('lightgbm', lgbm_params)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
    model.fit(x_train, y_train)
    print(model.get_metric(x_test, y_test))




if __name__ == '__main__':
    feature_df = load_general_feature_file()
