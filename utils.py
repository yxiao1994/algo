# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#
from datetime import timedelta
from evaluation import *
import torch.nn.functional as F


def weighted_loss(outputs, labels):
    #print(outputs, labels)
    weights = [0.4, 0.3, 0.2, 0.1]
    loss = 0.0
    for i, weight in enumerate(weights):
        loss += weight * F.binary_cross_entropy(outputs[:, i], labels[:, i])
    return loss


def weighted_auc(labels, outputs, userids):
    weights = [0.4, 0.3, 0.2, 0.1]
    avg_auc = 0
    for i, weight in enumerate(weights):
        avg_auc += weight * uAUC(labels[:, i], outputs[:, i], userids)
    return avg_auc


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def save_pred_file(config, test_prob):
    """保存预测结果"""
    test_data = pd.read_csv(config.test_path, encoding='utf-8')
    test_data = test_data[['userid', 'feedid']]
    assert len(test_data) == len(list(test_prob))
    test_data['read_comment'] = test_prob[:, 0]
    test_data['like'] = test_prob[:, 1]
    test_data['click_avatar'] = test_prob[:, 2]
    test_data['forward'] = test_prob[:, 3]
    features = ['userid', 'feedid', 'read_comment', 'like', 'click_avatar', 'forward']
    test_data[features].to_csv(config.pred_path, index=False)
