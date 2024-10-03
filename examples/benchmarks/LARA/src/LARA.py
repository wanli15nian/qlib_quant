# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Text, Union

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger

import warnings
warnings.filterwarnings('ignore')

import imp
model_all_utils = imp.load_source('model_all_utils', '../../src/model_all_utils.py')
model_all = imp.load_source('model_all_utils', '../../src/model_all.py')

class LARA(Model):
    """LARA Model"""

    def __init__(
        self,
        use_metric = 'True',
        use_multi_label = True,
        shuffle = True,
        preprocessing = 'None',
        metric_method = 'ITML_Supervised',
        ping = "True",
        knn_rnn = 'knn',
        points = 100,
        radius = 100,
        use_dual = True,
        dual_numbers = 5,
        dual_ratio = 0.01,
        **kwargs
    ):

        self.logger = get_module_logger("LARA")

        self.use_metric = use_metric
        self.use_multi_label = use_multi_label
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.metric_method = metric_method
        self.ping = ping
        self.knn_rnn = knn_rnn
        self.points = points
        self.radius = radius
        self.use_dual = use_dual
        self.dual_numbers = dual_numbers
        self.dual_ratio = dual_ratio

        self.params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': {'binary_logloss', 'auc'},
            'silent': 1
        }
        model_all_utils.set_seed(0, self.params)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )

        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_train, y_train = df_train["feature"], df_train["label"]
        x_test, y_test = df_test["feature"], df_test["label"]

        y_train['instrument'] = y_train.index.get_level_values('instrument')
        y_train['datetime'] = y_train.index.get_level_values('datetime')
        y_test['instrument'] = y_test.index.get_level_values('instrument')
        y_test['datetime'] = y_test.index.get_level_values('datetime')

        self.usefv = x_train.columns
        self.out = y_train.columns[0]
        
        print(x_valid.shape[0], 
              x_train.shape[0], 
              x_test.shape[0])
        print(y_valid[y_valid[self.out] > 0.001].shape[0], 
              y_train[y_train[self.out] > 0.001].shape[0], 
              y_test[y_test[self.out] > 0.001].shape[0])

        df_train = pd.concat([x_train[self.usefv], y_train], axis=1)
        df_test = pd.concat([x_test[self.usefv], y_test], axis=1)

        df_train, df_train_ping, train_label_metric_learning = self.data_processing(df_train)
        df_test = self.data_processing(df_test, False)

        train_dataset = (df_train[self.usefv], df_train[[self.out, 'label', 'instrument', 'datetime']])
        test_dataset = (df_test[self.usefv], df_test[[self.out, 'label', 'instrument', 'datetime']])
        ping_dataset = (df_train_ping[self.usefv], df_train_ping[[self.out, 'label', 'instrument', 'datetime']])

        if self.use_metric == "True":
            x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform, self.metric_model = model_all.metric_learning(self, train_dataset, test_dataset, ping_dataset, train_label_metric_learning)
        else:
            x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform \
            = train_dataset[0], train_dataset[1], test_dataset[0], test_dataset[1], ping_dataset[0], ping_dataset[1]

        self.logger.info("Complete Metric Learning")

        if self.ping == "True":
            x_train_transform = pd.concat(
                [x_train_transform, x_ping_transform], axis=0)
            y_train_transform = pd.concat(
                [y_train_transform, y_ping_transform], axis=0)
            x_train_transform.index = np.arange(x_train_transform.shape[0])
            y_train_transform.index = np.arange(y_train_transform.shape[0])
        
        if self.knn_rnn == "None":
            pass
        else:
            ann_path = os.path.join('./', 'ann_graph.bin')
            x_train_transform, y_train_transform, x_test_transform, y_test_transform = model_all.KNN(
                x_train_transform, y_train_transform, self, self.params, x_test_transform, y_test_transform, ann_path, train_test='train')

        test_date = y_test_transform[['instrument', 'datetime']]
        test_date.index = np.arange(test_date.shape[0])

        print(y_train_transform.head(10))

        self.logger.info("Complete KNN")

        bst = lgb.train(self.params, 
                    lgb.Dataset(x_train_transform, y_train_transform['label'])) 
        
        if self.use_dual:

            changes, gbm_list, predicted_df = model_all_utils.dual_label_one(x_train_transform, 
                                                        y_train_transform, 
                                                        x_test_transform, 
                                                        y_test_transform, 
                                                        bst, 
                                                        self, 
                                                        self.params, 
                                                        self.usefv)

            predicted_copy = predicted_df.copy()
            predicted_copy['signal'] = predicted_df.mean(1)
            predicted_copy.index = pd.MultiIndex.from_frame(test_date[['datetime', 'instrument']])

            self.predicted = predicted_copy

        else:
            self.predicted = pd.DataFrame(bst.predict(x_test_transform), columns=['signal'], index=pd.MultiIndex.from_frame(test_date[['datetime', 'instrument']]))

    def data_processing(self, df_train, train=True):
        df_train['label'] = 0
        df_train.loc[df_train[self.out] <= 0.001, 'label'] = 0
        df_train.loc[df_train[self.out] >= 0.001, 'label'] = 1

        if train == False:
            return df_train

        metric_learning_label = pd.DataFrame([])
        metric_learning_label[self.out] = df_train[self.out].copy()
        metric_learning_label['label'] = [0]*metric_learning_label.shape[0]
        metric_learning_label.loc[metric_learning_label[self.out]>= 1e-3, 'label'] = 5
        metric_learning_label.loc[(metric_learning_label[self.out] < 1e-3) & (metric_learning_label[self.out] >= 5e-4), 'label'] = 4
        metric_learning_label.loc[(metric_learning_label[self.out] < 5e-4) & (metric_learning_label[self.out] >= 0), 'label'] = 3
        metric_learning_label.loc[(metric_learning_label[self.out] < 0) & (metric_learning_label[self.out] >= -5e-4), 'label'] = 2
        metric_learning_label.loc[(metric_learning_label[self.out] < -5e-4) & (metric_learning_label[self.out] >= -1e-3), 'label'] = 1
        metric_learning_label.loc[metric_learning_label[self.out] < -1e-3, 'label'] = 0

        df_train, df_train_ping, train_label_metric_learning = model_all_utils.sample_equal(df_train, metric_learning_label, pcteject=0.001, out=self.out)

        return df_train, df_train_ping, train_label_metric_learning

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        
        return self.predicted['signal']
