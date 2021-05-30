# coding:utf-8
import torch
import sys
import time
import torch
from torch import nn
import numpy as np
from train_eval import train, test, init_network
from importlib import import_module
import argparse
import pandas as pd
import pickle
from data_loader import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from feature_config import MULTI_ID_FEATURES

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--mode', type=str, required=True, help='choose mode: train or test')
args = parser.parse_args()


def process_df(df):
    for f in MULTI_ID_FEATURES:
        df[f] = df[f].fillna('')
    df['manual_keyword_list'] = df['manual_keyword_list'].apply(lambda x: x.replace(';', ' '))
    df['manual_tag_list'] = df['manual_tag_list'].apply(lambda x: x.replace(';', ' '))
    return df


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 数据集
    dataset = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/algo/data'   # 数据集
    vocab = pickle.load(open(dataset + '/wechat_algo_data1/vocab_dic.txt', 'rb'))

    mode = args.mode    # train or test mode
    model_name = args.model  # bert

    x = import_module('models.' + model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = x.Config(dataset, vocab)
    model = x.Model(config).to(config.device)
    model = nn.DataParallel(model)

    if mode != 'test':
        #data = pd.read_csv(dataset + '/wechat_algo_data1/sample_data.csv')
        data = pd.read_csv(dataset + '/wechat_algo_data1/train_data.csv')
        data = process_df(data)
        train_data = data[data['date_'] < 14]
        dev_data = data[data['date_'] == 14]
        train_data = train_data.reset_index(drop=True)
        dev_data = dev_data.reset_index(drop=True)
        print(train_data.shape, dev_data.shape)
        train_dataset = TextDataset(vocab, train_data, config.device)
        dev_dataset = TextDataset(vocab, dev_data, config.device)

        train_sampler = RandomSampler(train_dataset)
        train_iter = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size,
                                      num_workers=4)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_iter = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config.eval_batch_size)
        train(config, model, train_iter, dev_iter)

    else:
        test_data = pd.read_csv(dataset + '/wechat_algo_data1/test_data.csv')
        test_data = process_df(test_data)
        print(test_data.shape)
        test_dataset = TextDataset(vocab, test_data, config.device)
        test_sampler = SequentialSampler(test_dataset)
        test_iter = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.eval_batch_size)
        test(config, model, test_iter)


