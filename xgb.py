# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#

from feature_config import *
from extrat_features import DATASET_PATH
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb
from evaluation import *

TRAIN_DATA_PATH = os.path.join(DATASET_PATH, "train_data.csv")
TEST_DATA_PATH = os.path.join(DATASET_PATH, "test_data.csv")
OUTPUT_PATH = os.path.join(DATASET_PATH, "xgb_predict.csv")
EMB_DIM = 100

params = {
        'booster': "gbtree",
        'objective': "binary:logistic",
        'eval_metric': "auc",
        'max_depth': 5,
        'subsample': 0.9,
        # 'colsample_bytree ': 0.7,
        'eta': 0.12,
        'missing': -999,
        # 'lambda': 3,
        # 'scale_pos_weight': 8
    }



def add_emb(df, feature, emb_path):
    new_features = []
    emb_dic = pickle.load(open(emb_path, 'rb'))
    embedding_matrix = np.zeros((len(df), EMB_DIM))
    embedding_matrix[:] = np.nan
    for i, key in enumerate(df[feature].values):
        key = int(key)
        if key in emb_dic:
            embedding_matrix[i] = emb_dic[key]
    for i in range(EMB_DIM):
        f = '_'.join([feature, 'emb', str(i)])
        new_features.append(f)
        df[f] = embedding_matrix[:, i]
    return new_features


def add_text_emb(df, feature, emb_path):
    new_features = []
    emb_dic = pickle.load(open(emb_path, 'rb'))
    embedding_matrix = np.zeros((len(df), EMB_DIM))
    embedding_matrix[:] = np.nan
    for i, key in enumerate(df['feedid'].values):
        key = int(key)
        if key in emb_dic:
            embedding_matrix[i] = emb_dic[key]
    for i in range(EMB_DIM):
        f = '_'.join([feature, 'emb', str(i)])
        new_features.append(f)
        df[f] = embedding_matrix[:, i]
    return new_features

    
def process_data(df):
    for feature in SINGLE_ID_FEATURES + ['device']:
        df[feature] = LabelEncoder().fit_transform(df[feature].apply(str))
    return df


if __name__ == '__main__':
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    df = pd.concat([train_data, test_data])
    res = test_data[['userid', 'feedid']]

    emb_feature_list = []
    feedid_emb_path = '/mnt/wfs/mmcommwfssz/user_hoonghu/algo/data/feed_pca_100d.pkl'
    emb_feature_list += add_emb(df, 'feedid', feedid_emb_path)
    description_emb_path = '/mnt/wfs/mmcommwfssz/user_hoonghu/algo/data/description.pkl'
    emb_feature_list += add_text_emb(df, 'description', description_emb_path)
    ocr_emb_path = '/mnt/wfs/mmcommwfssz/user_hoonghu/algo/data/ocr.pkl'
    emb_feature_list += add_text_emb(df, 'ocr', ocr_emb_path)
    asr_emb_path = '/mnt/wfs/mmcommwfssz/user_hoonghu/algo/data/asr.pkl'
    emb_feature_list += add_text_emb(df, 'asr', asr_emb_path)
    
    df = process_data(df)
    for action in ['read_comment', 'like', 'click_avatar', 'forward']:
        use_features = SINGLE_ID_FEATURES + DENSE_FEATURES + emb_feature_list
        df_train = df[df['date_'] < 14]
        df_val = df[df['date_'] == 14]
        df_test = df[df['date_'] == 15]
        train_x, val_x, test_x = df_train[use_features], df_val[use_features], df_test[use_features]
        train_y, val_y = df_train[action], df_val[action]
        xgtrain = xgb.DMatrix(train_x, label=train_y, feature_names=use_features)
        xgval = xgb.DMatrix(val_x, label=val_y, feature_names=use_features)
        xgtest = xgb.DMatrix(test_x, feature_names=use_features)

        watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        num_rounds = 1000

        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=3)
        preds = model.predict(xgtest)
        dev_preds = model.predict(xgval)
        uauc = uAUC(val_y.values, dev_preds, list(df_val['userid']))
        print(action + ':' + str(uauc))
        res[action] = preds
        model.save_model(os.path.join(DATASET_PATH, "{}_xgb_model.csv".format(action)))
    res.to_csv(OUTPUT_PATH, index=False)

