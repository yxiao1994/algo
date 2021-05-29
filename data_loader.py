import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from feature_config import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class TextDataset(Dataset):
    def __init__(self, vocab, df, device):
        # 数据预处理
        self.label = df[['read_comment', 'like', 'click_avatar', 'forward']].values
        self.userid = df['userid'].values
        self.id_features = {}
        for f in SINGLE_ID_FEATURES + MULTI_ID_FEATURES:
            self.id_features[f] = df[f].values
        self.dense_features = df[DENSE_FEATURES].values
        # self.embeddings_tables = []
        # for x in args.text_features:
        #     self.embeddings_tables.append(args.embeddings_tables[x[0]] if x[0] is not None else None)
        self.vocab = vocab
        self.max_len_text = 128
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        # 标签信息
        label = self.label[i]
        label = torch.LongTensor(label)
        userid = self.userid[i]
        userid = torch.LongTensor([userid])
        # 处理单值特征
        single_id_input = {}
        for f in SINGLE_ID_FEATURES:
            try:
                single_id_input[f] = self.vocab[f][str(self.id_features[f][i])]
            except:
                single_id_input[f] = self.vocab[f]['unk']

        # 处理多值特征
        multi_id_input, multi_id_mask = {}, {}
        for f in MULTI_ID_FEATURES:
            text_masks = np.zeros(self.max_len_text)
            text_ids = np.zeros(self.max_len_text, dtype=np.int64)
            x = self.id_features[f][i]
            for w_idx, word in enumerate(x.split()[:self.max_len_text]):
                text_masks[w_idx] = 1
                try:
                    text_ids[w_idx] = self.vocab[f][word]
                except:
                    text_ids[w_idx] = self.vocab[f]['unk']
            multi_id_input[f] = text_ids
            multi_id_mask[f] = text_masks

        # 浮点数特征
        dense_features = self.dense_features[i]
        dense_features = torch.FloatTensor(dense_features)

        single_id_concat = []
        for f in SINGLE_ID_FEATURES:
            single_id_concat.append(torch.LongTensor([single_id_input[f]]).view(-1, 1))
        single_id_concat = torch.cat(single_id_concat, dim=1)

        multi_id_concat = []
        for f in MULTI_ID_FEATURES:
            multi_id_concat.append(torch.LongTensor(multi_id_input[f]).view(-1, 1))
        multi_id_concat = torch.cat(multi_id_concat, dim=1)

        mask_concat = []
        for f in MULTI_ID_FEATURES:
            mask_concat.append(torch.FloatTensor(multi_id_mask[f]).view(-1, 1))
        mask_concat = torch.cat(mask_concat, dim=1)
        return (dense_features, single_id_concat, multi_id_concat, mask_concat, label, userid)
