import torch
import logging
import torch.nn as nn
import copy
import numpy as np
from feature_config import *
import os
from extrat_features import DATASET_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class Config(object):
    """配置参数"""

    def __init__(self, dataset, vocab):
        # self.model_name = 'rubbish_bert'
        self.model_name = 'baseline'
        self.test_path = dataset + '/wechat_algo_data1/test_data.csv'  # 测试集
        self.pred_path = dataset + '/wechat_algo_data1/predict_data.csv'  # 预测结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        # self.save_path = dataset + '/model_backup/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 500  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10  # epoch数
        self.train_batch_size = 1024   # mini-batch大小
        self.eval_batch_size = 1024  # mini-batch大小
        self.learning_rate = 2e-4  # 学习率
        self.embedding_size = 512
        self.vocab = vocab
        self.dropout_rate = 0.5
        self.class_num = 4


class EmbeddingLayer(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super().__init__()

        self.embed = nn.Embedding(feature_dim, embedding_dim, padding_idx=0)

        # normal weight initialization
        self.embed.weight.data.normal_(0., 0.0001)
        # TODO: regularization

    def forward(self, x):
        return self.embed(x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = list()
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        embedding_size = config.embedding_size
        self.feature_embedding_dict = dict()
        pre_feed_embedding = torch.from_numpy(np.load(os.path.join(DATASET_PATH, 'pretrained_feed_emb.npy')))
        feedid_embedding = nn.Embedding.from_pretrained(pre_feed_embedding, freeze=False)
        authorid_embedding = EmbeddingLayer(feature_dim=len(config.vocab['authorid']),
                                            embedding_dim=embedding_size)
        self.single_id_embedding_list = []
        for f in SINGLE_ID_FEATURES:
            if 'feedid' in f:
                self.single_id_embedding_list.append(feedid_embedding)
            elif 'authorid' in f:
                self.single_id_embedding_list.append(authorid_embedding)
            else:
                self.single_id_embedding_list.append(EmbeddingLayer(feature_dim=len(config.vocab[f]),
                                                                    embedding_dim=embedding_size))
        self.multi_id_embedding_list = []
        for f in MULTI_ID_FEATURES:
            if 'feedid' in f:
                self.multi_id_embedding_list.append(feedid_embedding)
            elif 'authorid' in f:
                self.multi_id_embedding_list.append(authorid_embedding)
            else:
                self.multi_id_embedding_list.append(EmbeddingLayer(feature_dim=len(config.vocab[f]),
                                                                   embedding_dim=embedding_size))
        self.single_id_embedding_list = nn.ModuleList(self.single_id_embedding_list)
        self.multi_id_embedding_list = nn.ModuleList(self.multi_id_embedding_list)
        fc_layer = FullyConnectedLayer(
            input_size=embedding_size * (len(SINGLE_ID_FEATURES) + len(MULTI_ID_FEATURES)) + len(DENSE_FEATURES),
            hidden_size=[200, 80, 1], bias=[True, True, False], activation='relu', sigmoid=True)
        self.output_layers = nn.ModuleList([
            copy.deepcopy(fc_layer)
            for _ in range(config.class_num)])

    def forward(self, dense_features, single_id_concat, multi_id_concat, mask_concat):
        feature_embedded = []
        for i, f in enumerate(SINGLE_ID_FEATURES):
            emb = self.single_id_embedding_list[i](single_id_concat[:, :, i].squeeze())
            feature_embedded.append(emb.float())

        for i, f in enumerate(MULTI_ID_FEATURES):
            embed = self.multi_id_embedding_list[i](multi_id_concat[:, :, i]).float()
            mask = mask_concat[:, :, i]
            embed_max = embed.max(1)[0].float()
            feature_embedded.append(embed_max)
            #print(embed)
            #print(mask)
            #embed_mean = (embed * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
            #feature_embedded.append(embed_mean.float())
        feature_embedded.append(dense_features.float())

        merged = torch.cat(feature_embedded, dim=1)
        res = []
        for layer in self.output_layers:
            res.append(layer(merged))
        #print(res)
        return torch.cat(res, dim=1)

