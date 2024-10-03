import os
import copy

import pandas as pd
import pickle

import argparse
parser = argparse.ArgumentParser(description='Hyper-paramters')
parser.add_argument('--train', action='store_true')
parser.add_argument('--dataset', type=str, default='Alpha158')
args = parser.parse_args()

thresh=0.001
count=1000

def train(dataset=args.dataset):
    base_path = os.path.join('experiments', dataset)
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)

    os.system('qrun ../../workflow_config_LARA_{}.yaml'.format(dataset))

def test(dataset=args.dataset):
    base_path = os.path.join('experiments', dataset)
    filenames = [os.path.join(path, file_dir)
                 for path, dir_list, file_list in os.walk(base_path)
                 for file_dir in dir_list if file_dir.find('artifacts') != -1]
    path = filenames[0]
    
    pred = pickle.load(open(os.path.join(path, 'pred.pkl'), 'rb'))
    label = pickle.load(open(os.path.join(path, 'label.pkl'), 'rb'))

    prec, wlr, ar = test_index(pred, label, up_or_down='up')   
 
    res = pd.DataFrame([[dataset, thresh, count, prec, wlr, ar]], \
               columns=['dataset', 'thresh', 'count', 'precision', 'win-loss ratio', 'average return'])
    print(res)
    

def test_index(pred, label, up_or_down):
    df = pd.concat([pred, label], axis=1, join='inner')
    name1, name2 = list(df.columns)

    assert up_or_down=='up', 'up_or_down must be "up" !'

    df['label'] = 0
    df = df.sort_values(by=[name1], ascending=False).iloc[:count]
    df.loc[df[name2] >= thresh, 'label'] = 1
    prec = df['label'].sum() / count

    win = df.loc[df[name2] > 0, name2].mean()
    loss = df.loc[df[name2] < 0, name2].mean()
    wlr = abs(win / loss)

    ar = df[name2].sum() / count

    return prec, wlr, ar

if __name__ == '__main__':
    if args.train:
        train(args.dataset)
    else:
        test(args.dataset)
