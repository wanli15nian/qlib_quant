import numpy as np
import pandas as pd
import logging
from os import makedirs as mkdir
import os
import random
import metric_learn
import json
from sklearn.cluster import KMeans
import scipy
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import *
# from index import backtest 
from pathos.multiprocessing import ProcessingPool as Pool

def set_logging(console_level, file_level, file):
    logging.basicConfig(filename=file, level=file_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger().addHandler(console)


def sample_equal(wdf, metric_learning_label, pcteject, out):
    traindf = wdf.loc[(abs(wdf[out]) >= pcteject), :]
    traindf_mid = wdf.loc[(abs(wdf[out]) < pcteject), :]
    train_metric_learning_label = metric_learning_label

    label0_num = traindf.loc[traindf['label'] == 0, :].shape[0]
    label1_num = traindf.loc[traindf['label'] == 1, :].shape[0]
    label2_num = traindf_mid.shape[0]

    if label0_num > label1_num:
        traindf = pd.concat([
            traindf.loc[traindf['label'] == 0, :].sample(label1_num),
            traindf.loc[traindf['label'] == 1, :]
        ])
        traindf_ping = traindf_mid.sample(min(label1_num, label2_num))
        traindf = traindf.sort_index()
        traindf_ping = traindf_ping.sort_index()
        train_metric_learning_label_df = pd.concat([
            train_metric_learning_label.loc[traindf.index],
            train_metric_learning_label.loc[traindf_ping.index]
        ],
                                                   axis=0)
    else:
        traindf = pd.concat([
            traindf.loc[traindf['label'] == 0, :],
            traindf.loc[traindf['label'] == 1, :].sample(label0_num)
        ])
        traindf_ping = traindf_mid.sample(min(label0_num, label2_num))
        traindf = traindf.sort_index()
        traindf_ping = traindf_ping.sort_index()
        train_metric_learning_label_df = pd.concat([
            train_metric_learning_label.loc[traindf.index],
            train_metric_learning_label.loc[traindf_ping.index]
        ],
                                                   axis=0)

    return traindf, traindf_ping, train_metric_learning_label_df


def data_preprocess(df, method, model=None):
    # Note：PowerTransformer：Note that Box-Cox can only be applied to strictly positive data.
    # But if negative values are present the Yeo-Johnson transformed is to be preferred.
    if method == 'StandardScaler':
        if model is None:
            processor = StandardScaler().fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'MinMaxScaler':
        if model is None:
            processor = MinMaxScaler().fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'MaxAbsScaler':
        if model is None:
            processor = MaxAbsScaler().fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'RobustScaler':
        if model is None:
            processor = RobustScaler(quantile_range=(25, 75)).fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'PowerTransformer':
        if model is None:
            processor = PowerTransformer(method='yeo-johnson').fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'QuantileTransformer_normal':
        if model is None:
            processor = QuantileTransformer(
                output_distribution='normal').fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'QuantileTransformer_uniform':
        if model is None:
            processor = QuantileTransformer(
                output_distribution='uniform').fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'Normalizer':
        if model is None:
            processor = Normalizer().fit(df)
            res_df = pd.DataFrame(processor.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df, processor
        else:
            res_df = pd.DataFrame(model.transform(df))
            res_df.columns = df.columns
            res_df.index = np.arange(df.shape[0])
            return res_df
    elif method == 'None':
        if model is None:
            return df, df
        else:
            return df
    else:
        raise RuntimeError('Unknown data preprocessing methods.')


def winsorize(trainx):
    mean = trainx.mean()
    std = trainx.std()
    trainx = (trainx - mean) / (std + 1e-6)
    std = std.replace(np.nan, 1e-6)
    trainx = trainx.clip(mean - 3 * std, mean + 3 * std, axis=1)
    return trainx


def addnoise(trainx, fvnoise_num, samplenoise):
    #trainx.index = range(trainx.shape[0])
    for i in range(fvnoise_num):
        trainx['noise_' + str(i)] = np.random.rand(trainx.shape[0]) * 3
    if samplenoise != 0:
        idx = np.random.choice(trainx.index,
                               int(trainx.shape[0] * samplenoise))
        for fv in trainx.columns:
            trainx.loc[idx, fv] = trainx.loc[idx, fv]+3 * \
                (np.random.rand(trainx.loc[idx, fv].shape[0])-0.5)
    return trainx


def train_test_split(wdf,
                     metric_learning_label,
                     fv,
                     out,
                     timespan,
                     pcteject=0,
                     equalsample=True,
                     normalize=False,
                     clip=None):
    trainstartdate, trainenddate, teststartdate, testenddate = timespan
    wdf['datetime'] = [
        item.split(' ')[0].replace('-', '') for item in wdf['date'].tolist()
    ]
    if trainstartdate == '':
        trainstartdate = wdf.date[0].split(' ')[0].replace('-', '')
    if testenddate == '':
        testenddate = wdf.date[-1].split(' ')[0].replace('-', '')
    wdf = wdf.dropna(subset=[out])

    metric_learning_label['datetime'] = [
        item.split(' ')[0].replace('-', '')
        for item in metric_learning_label['date'].tolist()
    ]
    if trainstartdate == '':
        trainstartdate = metric_learning_label.date[0].split(' ')[0].replace(
            '-', '')
    if testenddate == '':
        testenddate = metric_learning_label.date[-1].split(' ')[0].replace(
            '-', '')
    metric_learning_label = metric_learning_label.dropna(subset=[out])

    traindf, testdf, traindf_ping, train_metric_learning_label = sample_equal(
        wdf, metric_learning_label, trainstartdate, trainenddate,
        teststartdate, testenddate, pcteject, out)
    fv.remove(out)
    fv.remove('date')
    trainx = traindf.loc[:, fv]
    trainy = traindf.loc[:, [out, 'label', 'date']]
    testx = testdf.loc[:, fv]
    testy = testdf.loc[:, [out, 'label', 'date']]
    trainx_ping = traindf_ping.loc[:, fv]
    trainy_ping = traindf_ping.loc[:, [out, 'label', 'date']]
    train_metric_learning_label = train_metric_learning_label.loc[:, [
        out, 'label', 'date'
    ]]
    if normalize:
        trainx = winsorize(trainx)
        meanx = trainx.mean()
        stdx = trainx.std()
        trainx = (trainx - meanx) / (1e-4 + stdx)
        trainx.fillna(0, inplace=True)
        testx = (testx - meanx) / (1e-4 + stdx)
        testx.fillna(0, inplace=True)
        # 这里需要args的参数传进来
        testx = testx.clip(clip[0], clip[1], axis=1)
    fv = [k for k in trainx.columns]
    print('train test split complete')
    return trainx, testx, trainy, testy, fv, trainx_ping, trainy_ping, train_metric_learning_label


def calc_pct(wdf, out):
    n = int(out[3:])
    wdf[out] = (wdf['close'].shift(-n) - wdf['close']) / wdf['close']
    return wdf


def load_fromcsv(usefv, factor_dir, args):
    timespan, fund = args.timespan, args.fund
    # load data whose columns are usefv
    datalist = []
    filelist = os.listdir(os.path.join(factor_dir, fund))
    filelist.sort()

    if timespan[0] == '':
        timespan[0] = filelist[0].split('.')[0].replace('-', '')
    if timespan[-1] == '':
        timespan[-1] = filelist[-1].split('.')[0].replace('-', '')

    p = Pool(40)

    def load_func(file):
        date = file[:-4]
        date = date.replace('-', '')
        if ((date >= timespan[0]) and
            (date <= timespan[1])) or ((date >= timespan[2]) and
                                       (date <= timespan[3])):
            df = pd.read_csv(os.path.join(os.path.join(factor_dir, fund,
                                                       file))).loc[:, usefv]
            if 'short_is_cummax' in usefv:
                df['short_is_cummax'] = df['short_is_cummax'].astype(float)
            if 'short_is_cummin' in usefv:
                df['short_is_cummin'] = df['short_is_cummin'].astype(float)
            df['else_date'] = pd.to_datetime(df['else_date'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
            return df, file
        return None
    datalist = p.map(load_func, filelist)
    datalist = [item for item in datalist if not (item is None)]
    datalist = sorted(datalist, key=lambda x: x[1])
    datalist = [item[0] for item in datalist]

    df_all = pd.concat(datalist, axis=0)
    df_all = df_all.rename(columns={
        'return_' + args.out: args.out,
        'else_date': 'date'
    })
    
    df_all.index = np.arange(df_all.shape[0])
    print("Long factors: Complete")

    return df_all

def load_lrlist(lr_max, lr_min, num_rounds):
    lrlist = [
        lr_max + (lr_min - lr_max) * (np.log(i) / np.log(num_rounds))
        for i in range(1, num_rounds + 1)
    ]
    return lrlist


# setting sed
def set_seed(num, params):
    np.random.seed(num)
    random.seed(num)
    params['random_state'] = num
    params['seed'] = num
    params['feature_fraction_seed'] = num
    params['bagging_seed'] = num
    params['deterministic'] = True
    os.environ['PYTHONHASHSEED'] = str(num)


def cal(df, thresh):
    P = df[df["label"] == 1]
    N = df[df["label"] == 0]
    TP = P[P["proba"] > thresh].shape[0]
    FP = N[N["proba"] > thresh].shape[0]
    df_new = np.where(df.proba > thresh, 1, 0)

    f1_macro = f1_score(np.array(df['label']), df_new, average='macro')
    f1_micro = f1_score(np.array(df['label']), df_new, average='micro')
    f1_weighted = f1_score(np.array(df['label']), df_new, average='weighted')
    try:
        auc = metrics.roc_auc_score(np.array(df['label']),
                                    np.array(df['proba']))
    except:
        auc = 1

    if TP + FP <= 10:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    return (np.round(TP / (TP + FP), 4), np.round(TP / P.shape[0], 4),
            np.round((df_new == df['label']).sum() / df.shape[0],
                     4), P.shape[0], df[df["proba"] > thresh].shape[0],
            np.round(f1_macro, 4), np.round(f1_micro,
                                            4), np.round(f1_weighted,
                                                         4), np.round(auc, 4))


def save_param(dic, score, args):
    # filename="best_param_{}_dual_{}_{}.json".format(args.fund, args.begin_thresh, args.mode)
    filename = "best_param_{}.json".format(args.fund)
    dic["precision"] = score
    tmp = dic.pop('metric')
    with open(os.path.join('json', filename), "w") as f:
        f.write(json.dumps(dic, indent=2))
    dic['metric'] = tmp


def load_param(args, params):
    filename = "best_param_{}.json".format(args.fund)
    with open(os.path.join('json', filename), "r") as f:
        dic = json.load(f)
        dic['metric'] = params['metric']
    return dic


# Metric Learning from Weaker Supervision
def create_constraints(y):
    import itertools

    # aggregate indices of same class
    zeros = np.where(y == 0)[0]
    ones = np.where(y == 1)[0]
    twos = np.where(y == 2)[0]
    # make permutations of all those points in the same class
    zeros_ = list(itertools.combinations(zeros, 2))
    ones_ = list(itertools.combinations(ones, 2))
    twos_ = list(itertools.combinations(twos, 2))
    # put them together!
    sim = np.array(zeros_ + ones_ + twos_)

    # similarily, put together indices in different classes
    dis = []
    for zero in zeros:
        for one in ones:
            dis.append((zero, one))
        for two in twos:
            dis.append((zero, two))
    for one in ones:
        for two in twos:
            dis.append((one, two))

    # pick up just enough dissimilar examples as we have similar examples
    dis = np.array(random.sample(dis, len(sim)))

    # return an array of pairs of indices of shape=(2*len(sim), 2), and the
    # corresponding labels, array of shape=(2*len(sim))
    # Each pair of similar points have a label of +1 and each pair of
    # dissimilar points have a label of -1
    return (np.vstack([
        np.column_stack([sim[:, 0], sim[:, 1]]),
        np.column_stack([dis[:, 0], dis[:, 1]])
    ]), np.concatenate([np.ones(len(sim)), -np.ones(len(sim))]))


def perform_nan_inf(x, y):
    x = x.replace([np.inf, -np.inf], np.nan)
    x.index = np.arange(x.shape[0])
    y.index = np.arange(y.shape[0])
    x_new = x.dropna(axis=0)
    y_new = y.loc[x_new.index]
    x_new.index = np.arange(x_new.shape[0])
    y_new.index = np.arange(y_new.shape[0])
    return x_new, y_new


def get_columns(fund, factor_dir):
    filenames = os.listdir(os.path.join(factor_dir, fund))
    df = pd.read_csv(os.path.join(factor_dir, fund, filenames[0]))
    short_list, long_list, micro_list, stock_list, return_list, else_list = [], [], [], [], [], []
    for item in list(df.columns):
        if 'short' in item:
            short_list.append(item)
        elif 'long' in item:
            long_list.append(item)
        elif 'micro' in item:
            micro_list.append(item)
        elif 'stock' in item:
            stock_list.append(item)
        elif 'return' in item:
            return_list.append(item)
        elif 'else' in item:
            else_list.append(item)
    return short_list, long_list, micro_list, stock_list, return_list, else_list


def dual_label_one(train_data_in, train_label_in, test_data_in, test_label_in,
                   gbm, self, params, usefv):
    set_seed(3, params)
    train_data_new = train_data_in.copy()
    train_label_new = train_label_in.copy()
    gbm_list = []
    threshes_list1 = []
    threshes_list2 = []
    changes = []

    for index in range(self.dual_numbers):
        assert list(train_data_new.columns) == list(usefv)
        train_label_new['proba'] = gbm.predict(train_data_new)
        train_label_new['label1'] = 0
        train_label_new['label2'] = 0

        ratio = self.dual_ratio
        temp = train_label_new['proba'].copy()


        num1, num2 = int(
            (1 - ratio) * temp.shape[0]), int(ratio * temp.shape[0])
        idx1 = temp.sort_values(ascending=False).iloc[:num1].index
        idx2 = temp.sort_values(ascending=False).iloc[:num2].index

        train_label_new.loc[idx1, 'label1'] = 1
        train_label_new.loc[idx2, 'label2'] = 1
        temp_label = (
            train_label_new['label'] & train_label_new['label1']) | (
                (1 - train_label_new['label']) & train_label_new['label2'])

        change_number = sum(train_label_new['label'] ^ temp_label)
        changes.append(change_number)

        # renew the label
        train_label_new['label'] = temp_label

        # record the old model and train a new model
        gbm_list.append(gbm)
        lgb_train = lgb.Dataset(train_data_new, train_label_new["label"])
        gbm = lgb.train(params, lgb_train)

    actual_label_out = test_label_in.copy()
    predicted_df = pd.DataFrame([])
   
    # ensemble all the models 
    for index, gbm_instance in enumerate(gbm_list):
        temp = gbm_instance.predict(test_data_in)

        predicted_df = pd.concat([
            predicted_df,
            pd.DataFrame(temp, columns=['model' + str(index)])
        ], axis=1)

    return np.mean(changes), gbm_list, predicted_df
