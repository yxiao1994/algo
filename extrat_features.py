# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#
# coding: utf-8
import os
import time
from sklearn.utils import shuffle
import numpy as np
import logging
from itertools import combinations
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)
import pandas as pd
import math

# 存储数据的根目录
ROOT_PATH = "/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/algo/data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
# 训练集
USER_ACTION_PATH = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO_PATH = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE_PATH = os.path.join(DATASET_PATH, "test_a.csv")
VOCAB_DATA_PATH = os.path.join(DATASET_PATH, "vocab_data.csv")

LABEL_START_DAY = 8
END_DAY = 15
SEED = 2021
SAMPLE_RATE = 0.3
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]


def static_feature(history_data, dim, start_day=1, before_day=7, agg=['sum', 'mean']):
    """
    统计用户/feed 过去n天各类行为的次数
    :param history_data: DataFrame. 用于统计的数据
    :param dim: String. 待统计的特征
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    print('now process static feature:' + dim)
    user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
    res_arr = []
    for start in range(start_day, END_DAY - before_day + 1):
        x = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
        x = x.drop('date_', axis=1)
        temp = x.groupby([dim]).agg(agg).reset_index()
        features = [dim + '_' + '_'.join(x) for x in temp.columns.values]
        features[0] = dim
        temp.columns = features
        temp["date_"] = start + before_day
        res_arr.append(temp)
    dim_feature = pd.concat(res_arr)
    return dim_feature


def cross_feature(history_data, f1, f2, start_day=1, before_day=7, agg=['count', 'sum', 'mean']):
    """
    统计交叉特征
    :param history_data: DataFrame. 用于统计的数据
    :param f1: String. 待统计的特征
    :param f2: String. 待统计的特征
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    print('now process cross feature:{}_{}'.format(f1, f2))
    user_data = history_data[[f1, f2, "date_"] + FEA_COLUMN_LIST]
    res_arr = []
    for start in range(start_day, END_DAY - before_day + 1):
        x = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
        x = x.drop('date_', axis=1)
        temp = x.groupby([f1, f2]).agg(agg).reset_index()
        dim = f1 + '_' + f2 + '_cross'
        features = [dim + '_' + '_'.join(x) for x in temp.columns.values]
        features[0] = f1
        features[1] = f2
        temp.columns = features
        temp["date_"] = start + before_day
        res_arr.append(temp)
    dim_feature = pd.concat(res_arr)
    return dim_feature


def user_action_sequence(history_data, f1, f2, start_day=1, before_day=7):
    """
    统计用户 过去n天发生行为的序列
    :param history_data: DataFrame. 用于统计的数据
    :param f1: String. 行为，例如read_comment、like
    :param f2: String. 序列特征，例如feedid、authorid
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    """
    print('now process user sequence feature:' + f1 + '_' + f2)
    user_data = history_data[['userid', f1, f2, "date_"]]
    # if positive:
    user_data = user_data[user_data[f1] > 0]
    # else:
    #     user_data = user_data[user_data[f1] == 0]
    res_arr = []
    for start in range(start_day, END_DAY - before_day + 1):
        log = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
        log = log.drop('date_', axis=1)

        dic, items = {}, []
        for item in log[['userid', f2]].values:
            if item[1] is None:
                continue
            try:
                dic[item[0]].append(str(item[1]))
            except:
                dic[item[0]] = [str(item[1])]
        for key in dic:
            items.append([key, ' '.join(dic[key])])
        # 赋值序列特征
        temp = pd.DataFrame(items)
        f_name = '_'.join(['user', f1, f2, 'sequence'])
        temp.columns = ['userid', f_name]
        temp["date_"] = start + before_day
        res_arr.append(temp)
    seq_feature = pd.concat(res_arr)
    return seq_feature


def feed_action_sequence(history_data, f1, f2, start_day=1, before_day=7):
    """
    统计过去n天对feedid/authorid发生行为的用户序列
    :param history_data: DataFrame. 用于统计的数据
    :param f1: String. 行为，例如read_comment、like
    :param f2: String. 序列特征，例如feedid、authorid
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    """
    print('now process feed sequence feature:' + f1 + '_' + f2)
    user_data = history_data[['userid', f1, f2, "date_"]]
    # if positive:
    user_data = user_data[user_data[f1] > 0]
    # else:
    #     user_data = user_data[user_data[f1] == 0]
    res_arr = []
    for start in range(start_day, END_DAY - before_day + 1):
        log = user_data[(user_data["date_"] >= start) & (user_data["date_"] < (start + before_day))]
        log = log.drop('date_', axis=1)

        dic, items = {}, []
        for item in log[[f2, 'userid']].values:
            if item[1] is None:
                continue
            try:
                dic[item[0]].append(str(item[1]))
            except:
                dic[item[0]] = [str(item[1])]
        for key in dic:
            items.append([key, ' '.join(dic[key])])
        # 赋值序列特征
        temp = pd.DataFrame(items)
        f_name = '_'.join([f1, f2, 'user', 'sequence'])
        temp.columns = [f2, f_name]
        temp["date_"] = start + before_day
        res_arr.append(temp)
    seq_feature = pd.concat(res_arr)
    return seq_feature


def feature_idf(history_data, feature):
    """
    计算特征的idf
    :param history_data: DataFrame. 用于统计的数据
    :param feature: String. 例如userID/authorid/feedid
    :return:dic. 每个id的idf值
    """
    dic = {}
    for val in history_data[feature].values:
        dic[val] = dic.get(val, 0) + 1

    file_num = len(history_data)
    idf_dic = {}
    for word in dic:
        if dic[word] >= 5:
            idf_dic[word] = math.log10(file_num * 1.0 / dic[word])
    return idf_dic


def user_repensentation(history_data):
    """
    通过历史行为序列计算用户之间的各个维度相似度
    :param history_data: DataFrame. 用于统计的数据
    :return:dic. 用户行为过的序列
    """

    log = history_data[['userid', 'feedid', 'has_action']]
    log = log[log['has_action'] > 0]

    dic, items = {}, []
    for item in log[['userid', 'feedid']].values:
        if item[1] is None:
            continue
        try:
            dic[item[0]].add(item[1])
        except:
            dic[item[0]] = set([item[1]])

    return dic


def get_user_similarity(user1, user2, item_dic, idf_dic):
    if user1 not in item_dic or user2 not in item_dic:
        return [0, 0, 0]
    inter = item_dic[user1] & item_dic[user2]
    n1 = len(item_dic[user1])
    n2 = len(item_dic[user2])
    common_count = len(inter)
    common_rate = common_count / max(n1, n2)
    common_idf = 0
    for feedid in inter:
        common_idf += idf_dic.get(feedid, 3.0)
    return [common_count, common_rate, common_idf]


def get_sequence_similarity(ad_user, user_list, item_dic, idf_dic):
    count_list, rate_list, idf_list = [], [], []
    try:
        for user in user_list.split():
            user = int(user)
            common_count, common_rate, common_idf = \
                get_user_similarity(ad_user, user, item_dic, idf_dic)
            count_list.append(common_count)
            rate_list.append(common_rate)
            idf_list.append(common_idf)
    except:
        pass
    common_count_avg = np.mean(count_list) if len(count_list) > 0 else 0.0
    common_count_max = np.max(count_list) if len(count_list) > 0 else 0.0
    common_count_sum = np.sum(count_list) if len(count_list) > 0 else 0.0

    common_rate_avg = np.mean(rate_list) if len(rate_list) > 0 else 0.0
    common_rate_max = np.max(rate_list) if len(rate_list) > 0 else 0.0
    common_rate_sum = np.sum(rate_list) if len(rate_list) > 0 else 0.0

    common_idf_avg = np.mean(idf_list) if len(idf_list) > 0 else 0.0
    common_idf_max = np.max(idf_list) if len(idf_list) > 0 else 0.0
    common_idf_sum = np.sum(idf_list) if len(idf_list) > 0 else 0.0
    return [common_count_avg, common_count_max, common_count_sum, \
            common_rate_avg, common_rate_max, common_rate_sum, \
            common_idf_avg, common_idf_max, common_idf_sum]


def sequence_ad_interaction(df, f1, f2, item_dic, idf_dic):
    """
    候选user和历史user行为序列之间的交互
    :param df: DataFrame. 用于统计的数据
    :param f1: String. 行为，例如read_comment、like
    :param f2: String. 序列特征，例如feedid、authorid
    :return:
    """
    f_name = '_'.join([f1, f2, 'user', 'sequence'])
    features = []
    for ad_user, user_list in tqdm(df[['userid', f_name]].values):
        features.append(get_sequence_similarity(ad_user, user_list, item_dic, idf_dic))
    features = np.array(features)
    feature_name_list = [
        'common_count_avg', 'common_count_max', 'common_count_sum',
        'common_rate_avg', 'common_rate_max', 'common_rate_sum',
        'common_idf_avg', 'common_idf_max', 'common_idf_sum',
    ]
    for i, f in enumerate(feature_name_list):
        df['_'.join([f1, f2, f])] = features[:, i]
    return df


def main():
    t = time.time()

    train_df = pd.read_csv(USER_ACTION_PATH)

    test_df = pd.read_csv(TEST_FILE_PATH)
    test_df['date_'] = END_DAY
    diff_columns = set(train_df.columns) - set(test_df.columns)
    for column in diff_columns:
        test_df[column] = -1
    for f in ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]:
        test_df[f] = np.random.randint(0, 2, size=[len(test_df)])
    features = train_df.columns
    df = pd.concat([train_df[features], test_df[features]])

    feed_info = pd.read_csv(FEED_INFO_PATH)
    df = pd.merge(df, feed_info, on='feedid', how='left')

    df[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    df[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        df[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    df["videoplayseconds"] = np.log(df["videoplayseconds"] + 1.0)
    df[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        df[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
    df['has_action'] = df['read_comment'] + df['like'] + df['click_avatar'] + df['favorite'] + \
                       df['forward'] + df['comment'] + df['follow']
    # 8-14天的数据用于构造标签
    data = df[df['date_'] >= LABEL_START_DAY]
    if not os.path.exists(VOCAB_DATA_PATH):
        data[data['date_'] < END_DAY].to_csv(VOCAB_DATA_PATH, index=False)

    # 统计类特征
    for dim in ['userid', 'feedid', 'authorid']:
        dim_feature = static_feature(df, dim)
        data = pd.merge(data, dim_feature, on=[dim, 'date_'], how='left')

    f1 = 'userid'
    for f2 in ['feedid', 'authorid']:
        dim_feature = cross_feature(df, f1, f2)
        data = pd.merge(data, dim_feature, on=[f1, f2, 'date_'], how='left')

    # 用户行为序列
    for f1 in FEA_COLUMN_LIST + ['has_action']:
        for f2 in ['feedid', 'authorid']:
            seq_feature = user_action_sequence(df, f1, f2)
            data = pd.merge(data, seq_feature, on=['userid', 'date_'], how='left')

    for f1 in FEA_COLUMN_LIST + ['has_action']:
        for f2 in ['feedid', 'authorid']:
            seq_feature = feed_action_sequence(df, f1, f2)
            data = pd.merge(data, seq_feature, on=[f2, 'date_'], how='left')

    # user和序列交互
    history_data = df[df['date_'] < LABEL_START_DAY]
    item_dic = user_repensentation(history_data)
    feed_idf = feature_idf(history_data, 'feedid')
    for f1 in ['like', 'has_action']:
        data = sequence_ad_interaction(data, f1, 'feedid', item_dic, feed_idf)

    # 对数据做处理
    sum_feature = [f for f in data.columns if 'sum' in f
                   or 'max' in f or 'count' in f]
    for f in sum_feature:
        data[f] = data[f].fillna(0)
        data[f] = np.log(data[f] + 1.0)

    mean_feature = [f for f in data.columns if 'mean' in f
                    or 'avg' in f]
    for f in mean_feature:
        data[f] = data[f].fillna(0)

    train_data = data[data['date_'] < END_DAY]
    test_data = data[data['date_'] == END_DAY]

    test_data.to_csv(os.path.join(DATASET_PATH, "test_data.csv"), index=False)
    pos = train_data[train_data['has_action'] == 1]
    neg = train_data[train_data['has_action'] == 0]
    print(pos.shape, neg.shape)
    neg = neg.sample(frac=SAMPLE_RATE)
    print(neg.shape)
    df = shuffle(pd.concat([pos, neg]))
    print(df.shape)
    df.to_csv(os.path.join(DATASET_PATH, "train_data.csv"), index=False)
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == "__main__":
    main()
