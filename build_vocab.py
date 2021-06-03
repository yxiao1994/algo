# coding:utf-8
#
# Copyright 2019 Tencent Inc.
# Author: Yang Xiao(mlhustxiao@tecent.com)
#

from extrat_features import *
from collections import Counter
import pickle
from feature_config import FEATURE_MAP_DIC
FEED_EMBEDDING = '/mnt/wfs/mmcommwfssz/user_hoonghu/algo/data/feed_pca_100d.pkl'

def feature_vaocb(df, feature, feature_type):
    """
    建立词表
    :param df:
    :param feature:
    :param feature_type:
    :return:
    """
    print('bulid vocab:' + feature)
    dic = {}
    dic['pad'] = 0
    dic['unk'] = 1

    conter = Counter()
    for item in df[feature].values:
        item = str(item)
        if len(item) == 0:
            continue
        if feature_type == 'multi':
            for word in item.split():
                try:
                    conter[word] += 1
                except:
                    conter[word] = 1
        else:
            try:
                conter[item] += 1
            except:
                conter[item] = 1

    most_common = conter.most_common(100000)
    cont = 0
    for x in most_common:
        if x[1] >= 5:
            dic[x[0]] = len(dic)
            cont += 1
            if cont < 10:
                print(x[0], dic[x[0]])
    return dic


if __name__ == "__main__":
    df = pd.read_csv(VOCAB_DATA_PATH)

    features = ['description', 'description_char', 'ocr', 'ocr_char', 'asr', 'asr_char',
                'manual_keyword_list', 'manual_tag_list', 'bgm_song_id', 'bgm_singer_id']
    for f in features:
        df[f] = df[f].fillna('')
    df['manual_keyword_list'] = df['manual_keyword_list'].apply(lambda x: x.replace(';', ' '))
    df['manual_tag_list'] = df['manual_tag_list'].apply(lambda x: x.replace(';', ' '))

    multi_feature_list = ['description', 'ocr', 'asr',
                          'manual_keyword_list', 'manual_tag_list',
                          'description_char', 'ocr_char', 'asr_char']
    single_feature_list = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    vocab_dic = {}
    for f in multi_feature_list:
        vocab_dic[f] = feature_vaocb(df, f, 'multi')
    for f in single_feature_list:
        vocab_dic[f] = feature_vaocb(df, f, 'single')
    for f in FEATURE_MAP_DIC:
        vocab_dic[f] = vocab_dic[FEATURE_MAP_DIC[f]]

    feed_embedding_matrix = np.zeros((len(vocab_dic['feedid']), 100))
    emb_dic = pickle.load(open(FEED_EMBEDDING, 'rb'))     
    for feedid in emb_dic:
        emb = emb_dic[feedid]
        feedid = str(feedid)
        if feedid in vocab_dic['feedid']:
            key = vocab_dic['feedid'][feedid]
            feed_embedding_matrix[key] = emb
    np.save(os.path.join(DATASET_PATH, 'pretrained_feed_emb'), feed_embedding_matrix)
    pickle.dump(vocab_dic, open(os.path.join(DATASET_PATH, 'vocab_dic.txt'), 'wb'))

