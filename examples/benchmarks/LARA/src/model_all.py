from model_all_utils import *
import copy
import hnswlib
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from sklearn import neighbors
import time
from scipy.sparse import csc_matrix
import random
import yaml
import pickle


def run_train(params, args, config, train_dataset, test_dataset, usefv, ping_dataset, train_label_metric_learning, path):
    # make the dirs to save the model and signals.
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(path, 'signal'), exist_ok=True)

    train_dataset = (train_dataset[0][usefv], train_dataset[1])
    test_dataset = (test_dataset[0][usefv], test_dataset[1])
    ping_dataset = (ping_dataset[0][usefv], ping_dataset[1])

    if args.use_metric == "True":
        x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform \
            = metric_learning(train_dataset, test_dataset, args, ping_dataset, train_label_metric_learning)
    else:
        x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform \
            = train_dataset[0], train_dataset[1], test_dataset[0], test_dataset[1], ping_dataset[0], ping_dataset[1]

    return 

def run_test(params, args, config, train_dataset, test_dataset, usefv, ping_dataset, train_label_metric_learning, path):
    train_dataset = (train_dataset[0][usefv], train_dataset[1])
    test_dataset = (test_dataset[0][usefv], test_dataset[1])
    ping_dataset = (ping_dataset[0][usefv], ping_dataset[1])

    if args.use_metric == "True":
        x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform \
            = metric_learning(train_dataset, test_dataset, args, ping_dataset, train_label_metric_learning)
    else:
        x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_ping_transform, y_ping_transform \
            = train_dataset[0], train_dataset[1], test_dataset[0], test_dataset[1], ping_dataset[0], ping_dataset[1]

    if args.ping == "True":
        x_train_transform = pd.concat(
            [x_train_transform, x_ping_transform], axis=0)
        y_train_transform = pd.concat(
            [y_train_transform, y_ping_transform], axis=0)
        x_train_transform.index = np.arange(x_train_transform.shape[0])
        y_train_transform.index = np.arange(y_train_transform.shape[0])

    if config['knn_rnn'] == "None":
        pass
    else:
        ann_path = os.path.join(path, 'ann_graph.bin')
        x_train_transform, y_train_transform, x_test_transform, y_test_transform = KNN(
            x_train_transform, y_train_transform, args, config, params, x_test_transform, y_test_transform, ann_path)

    # get the date of the test set.
    test_date = y_test_transform['date']
    test_date.index = np.arange(test_date.shape[0])

    gbm_list = sorted(os.listdir(os.path.join(path, 'models')))
    predicted_df = pd.DataFrame([])
    for index, gbm_path in enumerate(gbm_list):
        gbm_instance = lgb.Booster(model_file=os.path.join(path, 'models', gbm_path))
        
        temp = gbm_instance.predict(x_test_transform)
        predicted_df = pd.concat([
            predicted_df,
            pd.DataFrame(temp, columns=['model' + str(index)])
        ], axis=1)

    # save the signal 
    predicted_copy = predicted_df.copy()
    predicted_copy['signal'] = 0
    predicted_copy.loc[predicted_df.mean(1) > config['{}min_thres'.format(args.timescale)], 'signal'] = 1
    df = pd.concat([test_date, predicted_copy], axis=1) 
    df.to_csv(os.path.join(path, 'signal', 'test-signal.csv'), index=False)
   
    return calculate_index(args, df)

def pre_processing():
    parser = argparse.ArgumentParser(description='Hyper-parameters')
    # Required parameters
    parser.add_argument('--train_test', type=str, default='train', help="Whether to train or test.")
    parser.add_argument('--fund', type=str, default='512480.SH', 
                        choices=['512480.SH', '512880.SH', '515050.SH', 'BTC'], help='Input ETF.')
    parser.add_argument('--factor_path', type=str, 
                        help='Input the path of ETF factors.')
    parser.add_argument('--factor_list', type=str, 
                        help='Input the list of ETF factors you use.')
    parser.add_argument('--timespan',type=str, nargs="+",
                        default=['20201225', '20210222', '20210223', '20210309'])   
                        # default=['20201225', '20210101', '20210102', '20210105'])   
    parser.add_argument('--timescale', type=int, default=1,
                        choices=[1, 5, 10, 30], help='Input time scale(minutes).') 
    parser.add_argument('--updown', type=str, default='up',
                        help='Predict up or down')
    # 
    parser.add_argument('--save_path', type=str, 
                        help='Input the list of ETF factors you use.')
    parser.add_argument('--load_path', type=str, 
                        help='Input the list of ETF factors you use.')          

    # Ad-hoc parameters
    parser.add_argument('--use_metric', type=str, default='True')
    parser.add_argument('--ping', type=str, default='True')
    parser.add_argument('--mode', type=str, default='dual-ratio',
                        help='Mode you can choose: dual-fixed, dual-floor, norm, dual-ratio')
    parser.add_argument('--ensemble_mode', type=str, default='and')
    parser.add_argument('--factors', type=str, default='all',
                        help='Factors you can choose: long, short, micro, stock, all')
    parser.add_argument('--out', type=str, default='PCT20-price')
    parser.add_argument('--pcteject', type=float, default=1e-3)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--preprocessing', type=str, default='None', help="Preprocessing feature methods, which includes: "
                        "StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer_normal, QuantileTransformer_uniform, Normalizer, None")
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether to shuffle the dataset.')
    parser.add_argument('--metric_method', type=str, default='SDML_Supervised', help="First choince: ITML_Supervised, SDML_Supervised, LSML_Supervised. "
                        "Second choice: LMNN, MMC_Supervised, NCA, LFDA, RCA_Supervised, MLKR, ITML, MLKR for regression, ITML for pairwise")
    parser.add_argument('--use_multi_label', type=str, default='True', 
                        help='Use multi-label sets.')
                        
    parser.set_defaults(use_metric='True',
                        ping='True',
                        mode='dual-ratio',
                        ensemble_mode='and',
                        factors='all',
                        out='PCT100-price',
                        pcteject=1e-3,
                        normalize=False,
                        preprocessing='None',
                        shuffle=True,
                        metric_method='SDML_Supervised',
                        use_multi_label='True',
                        )

    args = parser.parse_args()
    
    # parameters for lightgbm
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': {'binary_logloss', 'auc'},
        'silent': 1
    }

    # get attributes of the data
    short_list, long_list, micro_list, stock_list, return_list, else_list = get_columns(args.fund, args.factor_path)
    
    config_file = open('parameters/%s.yaml'%args.fund)
    config = yaml.load(config_file)
    return args, params, config, short_list, long_list, micro_list, stock_list, return_list, else_list


def data_processing(args, short_list, long_list, micro_list, stock_list, return_list, else_list):
    assert 'return_'+args.out in return_list
    assert 'else_date' in else_list
    temp_list = ['return_'+args.out, 'else_date']

    if args.factors == "all":
        usefv = short_list + micro_list + stock_list + temp_list
    elif args.factors == "short":
        usefv = short_list + temp_list
    elif args.factors == "long":
        usefv = long_list + temp_list
    elif args.factors == "micro":
        usefv = micro_list + temp_list
    elif args.factors == "stock":
        usefv = stock_list + temp_list

    # load data
    wdf = load_fromcsv(usefv, args.factor_path, args)
    wdf.index = np.arange(wdf.shape[0])
    print('complete loading data.')

    usefv.remove('return_'+args.out)
    usefv.append(args.out)
    usefv.remove('else_date')
    usefv.append('date')

    wdf['label'] = 1
    if args.updown == "up":
        wdf.loc[wdf[args.out] <= 0.001, 'label'] = 0
    elif args.updown == "down":
        wdf.loc[wdf[args.out] >= -0.001, 'label'] = 0
    # 分成6类
    metric_learning_label = pd.DataFrame([])
    metric_learning_label[args.out] = wdf[args.out].copy()
    metric_learning_label['date'] = wdf['date'].copy()
    metric_learning_label['label'] = [0]*metric_learning_label.shape[0]
    metric_learning_label.loc[metric_learning_label[args.out]
                              >= 1e-3, 'label'] = 5
    metric_learning_label.loc[(metric_learning_label[args.out] < 1e-3)
                              & (metric_learning_label[args.out] >= 5e-4), 'label'] = 4
    metric_learning_label.loc[(metric_learning_label[args.out] < 5e-4)
                              & (metric_learning_label[args.out] >= 0), 'label'] = 3
    metric_learning_label.loc[(metric_learning_label[args.out] < 0) & (
        metric_learning_label[args.out] >= -5e-4), 'label'] = 2
    metric_learning_label.loc[(metric_learning_label[args.out] < -5e-4)
                              & (metric_learning_label[args.out] >= -1e-3), 'label'] = 1
    metric_learning_label.loc[metric_learning_label[args.out]
                              < -1e-3, 'label'] = 0

    trainx, testx, trainy, testy, usefv, trainx_ping, trainy_ping, train_label_metric_learning = train_test_split(
        wdf, metric_learning_label, usefv, args.out, args.timespan, args.pcteject, args.normalize)

    return (trainx, trainy), (testx, testy), usefv, (trainx_ping, trainy_ping), train_label_metric_learning


def metric_learning(self, train_dataset, test_dataset, train_ping_dataset, train_label_metric_learning):
    x_train, y_train = train_dataset[0], train_dataset[1]
    x_test, y_test = test_dataset[0], test_dataset[1]
    x_train_ping, y_train_ping = train_ping_dataset[0], train_ping_dataset[1]

    if self.use_multi_label:
        x_train_new, y_train_new = perform_nan_inf(x_train, y_train)
        x_train_ping_new, y_train_ping_new = perform_nan_inf(
            x_train_ping, y_train_ping)

        x_metric_learning = pd.concat([x_train_new, x_train_ping_new], axis=0)
        y_metric_learning = train_label_metric_learning
        x_metric_learning.index = np.arange(x_metric_learning.shape[0])
        y_metric_learning.index = np.arange(y_metric_learning.shape[0])
        # shuffle
        if self.shuffle:
            x_metric_learning = x_metric_learning.sample(frac=1)
            y_metric_learning = y_metric_learning.loc[x_metric_learning.index]
            x_metric_learning.index = np.arange(x_metric_learning.shape[0])
            y_metric_learning.index = np.arange(y_metric_learning.shape[0])
        x_metric_learning, y_metric_learning = perform_nan_inf(
            x_metric_learning, y_metric_learning)
        x_metric_learning, model = data_preprocess(
            x_metric_learning, self.preprocessing, model=None)
        x_metric_learning, y_metric_learning = perform_nan_inf(
            x_metric_learning, y_metric_learning)
        assert x_metric_learning.isna().sum().sum() == 0 and np.isinf(x_metric_learning).sum().sum() == 0 and y_metric_learning.isna(
        ).sum().sum() == 0 and np.isinf(y_metric_learning.loc[:, [self.out, 'label']]).sum().sum() == 0

    else:
        x_train_new, y_train_new = perform_nan_inf(x_train, y_train)
        x_train_new, model = data_preprocess(
            x_train_new, self.preprocessing, model=None)
        x_train_new, y_train_new = perform_nan_inf(x_train_new, y_train_new)
        assert x_train_new.isna().sum().sum() == 0 and np.isinf(x_train_new).sum().sum() == 0 and y_train_new.isna(
        ).sum().sum() == 0 and np.isinf(y_train_new.loc[:, [self.out, 'label']]).sum().sum() == 0

        x_train_ping_new, y_train_ping_new = perform_nan_inf(
            x_train_ping, y_train_ping)
        x_train_ping_new = data_preprocess(
            x_train_ping_new, self.preprocessing, model)
        x_train_ping_new, y_train_ping_new = perform_nan_inf(
            x_train_ping_new, y_train_ping_new)
        assert x_train_ping_new.isna().sum().sum() == 0 and np.isinf(x_train_ping_new).sum().sum() == 0 and y_train_ping_new.isna(
        ).sum().sum() == 0 and np.isinf(y_train_ping_new.loc[:, [self.out, 'label']]).sum().sum() == 0

        x_metric_learning = x_train_new.copy()
        y_metric_learning = y_train_new.copy()

    x_test_new, y_test_new = perform_nan_inf(x_test, y_test)
    x_test_new = data_preprocess(x_test_new, self.preprocessing, model)
    x_test_new, y_test_new = perform_nan_inf(x_test_new, y_test_new)
    assert x_test_new.isna().sum().sum() == 0 and np.isinf(x_test_new).sum().sum() == 0 and y_test_new.isna(
    ).sum().sum() == 0 and np.isinf(y_test_new.loc[:, [self.out, 'label']]).sum().sum() == 0
    
    assert train_label_metric_learning.isna().sum().sum() == 0 and np.isinf(
        train_label_metric_learning.loc[:, [self.out, 'label']]).sum().sum() == 0

    # Metric leanring need normalization.
    if self.metric_method == 'LMNN':
        model = metric_learn.LMNN(k=5)
    elif self.metric_method == 'ITML_Supervised':
        model = metric_learn.ITML_Supervised(
            prior='covariance', num_constraints=1000)
    elif self.metric_method == 'SDML_Supervised':
        model = metric_learn.SDML_Supervised(
            sparsity_param=0.1, balance_param=0.001, num_constraints=1000)
    elif self.metric_method == 'LSML_Supervised':
        model = metric_learn.LSML_Supervised(
            prior='covariance', num_constraints=1000)
    elif self.metric_method == 'MMC_Supervised':
        model = metric_learn.MMC_Supervised()
    elif self.metric_method == 'NCA':
        model = metric_learn.NCA(max_iter=1000)
    elif self.metric_method == 'LFDA':
        model = metric_learn.LFDA(k=2, n_components=2)
    elif self.metric_method == 'RCA_Supervised':
        model = metric_learn.RCA_Supervised(num_chunks=30, chunk_size=2)
    elif self.metric_method == 'MLKR':
        model = metric_learn.MLKR()
    elif self.metric_method == 'ITML':
        model = metric_learn.ITML(preprocessor=x)

    if self.metric_method != 'MLKR' and self.metric_method != 'ITML':
        if self.metric_method == 'ITML_Supervised':
            try:
                model.fit(x_metric_learning, y_metric_learning['label'])
            except:
                model = metric_learn.ITML_Supervised(num_constraints=1000)
                model.fit(x_metric_learning, y_metric_learning['label'])
        elif self.metric_method == 'LSML_Supervised':
            try:
                model.fit(x_metric_learning, y_metric_learning['label'])
            except:
                model = metric_learn.LSML_Supervised(num_constraints=1000)
                model.fit(x_metric_learning, y_metric_learning['label'])
        else:
            try:
                model.fit(x_metric_learning, y_metric_learning['label'])
            except:
                # model = metric_learn.ITML_Supervised(num_constraints=1000)
                # model.fit(x_metric_learning, y_metric_learning['label'])
                raise RuntimeError('SDML error(Non positive definite matrix).')
        x_train_transform = model.transform(x_train_new)
        x_test_transform = model.transform(x_test_new)
        x_train_ping_transform = model.transform(x_train_ping_new)
    elif self.metric_method == 'MLKR':
        # 
        model.fit(x_metric_learning, y_metric_learning[self.out])
        x_train_transform = model.transform(x_train_new)
        x_test_transform = model.transform(x_test_new)
        x_train_ping_transform = model.transform(x_train_ping_new)
    elif self.metric_method == 'ITML':
        pairs, pairs_labels = create_constraints(y_metric_learning['label'])
        model.fit(pairs, pairs_labels)
        x_train_transform = model.transform(x_train_new)
        x_test_transform = model.transform(x_test_new)
        x_train_ping_transform = model.transform(x_train_ping_new)

    x_train_transform = pd.DataFrame(x_train_transform)
    x_train_transform.columns = x_train_new.columns
    x_train_transform.index = np.arange(x_train_transform.shape[0])
    assert x_train_transform.shape == x_train_new.shape
    
    x_test_transform = pd.DataFrame(x_test_transform)
    x_test_transform.columns = x_test_new.columns
    x_test_transform.index = np.arange(x_test_transform.shape[0])
    assert x_test_transform.shape == x_test_new.shape

    x_train_ping_transform = pd.DataFrame(x_train_ping_transform)
    x_train_ping_transform.columns = x_train_ping_new.columns
    x_train_ping_transform.index = np.arange(x_train_ping_transform.shape[0])
    assert x_train_ping_transform.shape == x_train_ping_new.shape

    y_train_transform = y_train_new.copy()
    y_test_transform = y_test_new.copy()
    y_train_ping_transform = y_train_ping_new.copy()

    x_train_transform.index = np.arange(x_train_transform.shape[0])
    y_train_transform.index = np.arange(y_train_transform.shape[0])
    assert x_train_transform.shape == x_train_new.shape and y_train_transform.shape == y_train_new.shape

    x_test_transform.index = np.arange(x_test_transform.shape[0])
    y_test_transform.index = np.arange(y_test_transform.shape[0])
    assert x_test_transform.shape == x_test_new.shape and y_test_transform.shape == y_test_new.shape

    x_train_ping_transform.index = np.arange(x_train_ping_transform.shape[0])
    y_train_ping_transform.index = np.arange(y_train_ping_transform.shape[0])
    assert x_train_ping_transform.shape == x_train_ping_new.shape and y_train_ping_transform.shape == y_train_ping_new.shape

    return x_train_transform, y_train_transform, x_test_transform, y_test_transform, x_train_ping_transform, y_train_ping_transform, model


def KNN(x_train_transform, y_train_transform, self, params, x_test_transform, y_test_transform, path, train_test='train'):
    x_train_transform.index = np.arange(x_train_transform.shape[0])
    y_train_transform.index = np.arange(y_train_transform.shape[0])
    x_test_transform.index = np.arange(x_test_transform.shape[0])
    y_test_transform.index = np.arange(y_test_transform.shape[0])

    y_return, y_label = y_train_transform[self.out], y_train_transform['label']
    y_return.index = np.arange(y_return.shape[0])
    y_label.index = np.arange(y_label.shape[0])

    def fit_hnsw_index(features, ef=100, M=16, save_index_file=False):
        # Convenience function to create HNSW graph
        # features : list of lists containing the embeddings
        # ef, M: parameters to tune the HNSW algorithm
        num_elements = len(features)
        labels_index = np.arange(num_elements)
        EMBEDDING_SIZE = len(features[0])    # Declaring index
        # possible space options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known
        p = hnswlib.Index(space='l2', dim=EMBEDDING_SIZE)
        p.init_index(max_elements=num_elements, random_seed=100,
                     ef_construction=ef, M=M)    # Element insertion
        # Controlling the recall by setting ef
        int_labels = p.add_items(features, labels_index)
        # ef should always be > k
        p.set_ef(ef)
        p.set_num_threads(20)
        # If you want to save the graph to a file
        if save_index_file:
            p.save_index(path)
            print(path)

        return p

    if train_test == 'train':
        model = fit_hnsw_index(x_train_transform.values.tolist(), ef=self.points*10)
    else:
        model = hnswlib.Index(space='l2', dim=x_train_transform.shape[1])
        model.load_index(path, max_elements=x_train_transform.shape[0])
        model.set_ef(self.points*10)
        model.set_num_threads(20)
    
    # for testing
    ann_neighbor_indices, ann_distances = model.knn_query(
        x_test_transform.values.tolist(), self.points)
    if self.knn_rnn == 'knn':
        row = np.array([[i] for i in range(x_test_transform.shape[0])])
        row = np.tile(row, (1, self.points))
        row = row.reshape(-1)
        col = ann_neighbor_indices.reshape(-1)
        data = np.array([1]*col.shape[0])
        arr = csc_matrix((data, (row, col)), shape=(
            x_test_transform.shape[0], x_train_transform.shape[0]))
    elif self.knn_rnn == 'rnn':
        row = np.array([[i] for i in range(x_test_transform.shape[0])])
        row = np.tile(row, (1, self.points))
        row = row.reshape(-1)
        col = ann_neighbor_indices.reshape(-1)
        ann_distances[ann_distances > self.radius] = 0
        ann_distances = 1 / ann_distances
        ann_distances[np.isinf(ann_distances)] = 0
        data = ann_distances.reshape(-1)
        arr = csc_matrix((data, (row, col)), shape=(
            x_test_transform.shape[0], x_train_transform.shape[0]))

    row = [i for i in range(y_train_transform.shape[0])]
    col = [0] * y_train_transform.shape[0]
    data = list(y_train_transform.copy()['label'].values)
    arr2 = csc_matrix((data, (row, col)), shape=(
        x_train_transform.shape[0], 1))
    arr2[arr2 < 1] = -1
    arr = csc_matrix.dot(arr, arr2).toarray().reshape(-1)
    x_test_transform = x_test_transform.loc[arr >= 0, :]
    y_test_transform = y_test_transform.loc[arr >= 0, :]
    x_test_transform.index = np.arange(x_test_transform.shape[0])
    y_test_transform.index = np.arange(y_test_transform.shape[0])
    if train_test == 'test':
        return x_train_transform, y_train_transform, x_test_transform, y_test_transform

    # for training
    ann_neighbor_indices, ann_distances = model.knn_query(
        x_train_transform.values.tolist(), self.points)
    if self.knn_rnn == 'knn':
        row = np.array([[i] for i in range(x_train_transform.shape[0])])
        row = np.tile(row, (1, self.points))
        row = row.reshape(-1)
        col = ann_neighbor_indices.reshape(-1)
        data = np.array([1]*col.shape[0])
        arr = csc_matrix((data, (row, col)), shape=(
            x_train_transform.shape[0], x_train_transform.shape[0]))
    elif self.knn_rnn == 'rnn':
        row = np.array([[i] for i in range(x_train_transform.shape[0])])
        row = np.tile(row, (1, self.points))
        row = row.reshape(-1)
        col = ann_neighbor_indices.reshape(-1)
        ann_distances[ann_distances > self.radius] = 0
        ann_distances = 1 / ann_distances
        ann_distances[np.isinf(ann_distances)] = 0
        data = ann_distances.reshape(-1)
        arr = csc_matrix((data, (row, col)), shape=(
            x_train_transform.shape[0], x_train_transform.shape[0]))

    row = [i for i in range(y_train_transform.shape[0])]
    col = [0] * y_train_transform.shape[0]
    data = list(y_train_transform.copy()['label'].values)
    arr2 = csc_matrix((data, (row, col)), shape=(
        x_train_transform.shape[0], 1))
    arr2[arr2 < 1] = -1
    arr = csc_matrix.dot(arr, arr2).toarray().reshape(-1)
    x_train_transform = x_train_transform.loc[arr >= 0, :]
    y_train_transform = y_train_transform.loc[arr >= 0, :]
    x_train_transform.index = np.arange(x_train_transform.shape[0])
    y_train_transform.index = np.arange(y_train_transform.shape[0])

    return x_train_transform, y_train_transform, x_test_transform, y_test_transform

def produce_feature_importance(train_dataset, ping_dataset, params):
    train_dataset = (train_dataset[0], train_dataset[1])
    ping_dataset = (ping_dataset[0], ping_dataset[1])

    x_train_transform, y_train_transform, x_ping_transform, y_ping_transform \
        = train_dataset[0], train_dataset[1], ping_dataset[0], ping_dataset[1]
    
    x_train_transform = pd.concat(
        [x_train_transform, x_ping_transform], axis=0)
    y_train_transform = pd.concat(
        [y_train_transform, y_ping_transform], axis=0)
    x_train_transform.index = np.arange(x_train_transform.shape[0])
    y_train_transform.index = np.arange(y_train_transform.shape[0])

    
    booster = lgb.train(params, 
              lgb.Dataset(x_train_transform, y_train_transform['label']))
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()
    res = []
    for (feature_name,importance) in zip(feature_name,importance):
        res.append([feature_name,importance])
    res = sorted(res, key=lambda x: x[1], reverse=True)
    with open('factor_list/feature_importance_BTC.txt', 'w') as f:
        for name, score in res:
            f.write(name + '\n')

def load_factorlist(factor_list):
    usefv = []
    with open(factor_list, "r") as f:
        lines = f.readlines()
        for line in lines:
            usefv.append(line.strip())
    usefv = usefv[:40]
    return usefv


if __name__ == "__main__":

    def main():
        args, params, config, short_list, long_list, micro_list, stock_list, return_list, else_list = pre_processing()
        
        if args.use_metric == 'False':
            args.ping = 'False'

        # train function
        def train(seed):
            # set random seed
            set_seed(seed, params)

            # get config
            path = os.path.join(args.save_path, 'seed-{}'.format(seed))
            train_dataset, test_dataset, usefv, ping_dataset, train_label_metric_learning = data_processing(
                args, short_list, long_list, micro_list, stock_list, return_list, else_list)

            # if not os.path.exists(args.factor_list):
            # produce_feature_importance(train_dataset, ping_dataset, params) 

            # get factor_list
            usefv = load_factorlist(args.factor_list)

            # run this model
            return run_train(params, args, config[args.updown], train_dataset, test_dataset, usefv,
                ping_dataset, train_label_metric_learning, path)

        def test(seed):
            # set random seed
            set_seed(seed, params)

            # get config
            path = os.path.join(args.load_path, 'seed-{}'.format(seed))
            train_dataset, test_dataset, usefv, ping_dataset, train_label_metric_learning = data_processing(
                args, short_list, long_list, micro_list, stock_list, return_list, else_list)
    
            # get factor_list
            usefv = load_factorlist(args.factor_list)

            # test this model
            return run_test(params, args, config[args.updown], train_dataset, test_dataset, usefv,
                ping_dataset, train_label_metric_learning, path)

        # process the random seed
        if isinstance(config['seed'], list):
            res = []
            f = open('error_{}.txt'.format(args.updown), 'w')
            for seed in config['seed']:
                try:
                    if args.train_test == 'train':
                        res.append(train(seed))
                    else:
                        res.append(test(seed))
                    f.write('succe seed: {}\n'.format(seed))
                except:
                    f.write('error seed: {}\n'.format(seed))
                f.flush()
                        
            # res = pd.concat(res, axis=0)
            # res = pd.DataFrame([res.mean(axis=0), res.std(axis=0)])
            # res.index = ['mean', 'std']
            # res.to_csv(os.path.join(args.save_path, 'index.csv'))
            # print(res)
        elif isinstance(config['seed'], int):
            if args.train_test == 'train':
                res = train(config['seed'])
            else:
                res = test(config['seed'])
            # res.to_csv(os.path.join(args.save_path, 'index.csv'))
            # print(res)
        else:
            raise RuntimeError('Unsupported Seed.')

    main()
