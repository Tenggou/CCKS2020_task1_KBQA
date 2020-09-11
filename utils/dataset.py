from itertools import permutations
import random

import torch
from torch.utils.data import Dataset

from model.transformers import BertTokenizer
from utils.configure import BERT_VOCAB


def test_collate(batch_data):
    return batch_data[0]


def train_collate(batch_data):
    # 将数据整理为能够并行处理的格式
    return {
        'input_ids': torch.cat(tuple([data['input_ids'] for data in batch_data])),
        'token_type_ids': torch.cat(tuple([data['token_type_ids'] for data in batch_data])),
        'label': torch.cat(tuple([data['label'] for data in batch_data]))
    }


class PairDataset(Dataset):
    def __init__(self, data, device, is_train, is_test=False):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)
        self.max_len = 64

        self.data = data
        self.device = device
        self.is_train = is_train
        self.is_test = is_test

    def __getitem__(self, item):
        question = self.data[item]['question']

        if not self.is_test:
            # 训练正例
            true_pair = []  # index
            true_pair_ = []  # 文字
            for p in self.data[item]['parse']:
                pair = p[:3]
                true_pair.append(''.join(pair))
                true_pair_.append(pair)

        if self.is_train:
            # 训练负例
            fake_pair = []
            for value in self.data[item]['pair_candidates'].values():
                for v in value:  # each entity_relation pair
                    if v not in true_pair_:
                        fake_pair.append(''.join(v))

            # 每个就选取负例来训练
            if len(fake_pair) > 64:
                fake_pair = random.sample(fake_pair, 64)

            label = [0] * len(fake_pair) + [1] * len(true_pair)

            input_ids = []
            token_type_ids = []
            for each in fake_pair + true_pair:
                enc = self.tokenizer.encode_plus(
                    question,
                    each,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
                input_ids.append(enc['input_ids'])
                token_type_ids.append(enc['token_type_ids'])
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
            }
        else:  # evaluate and test
            pair = []  # index
            pair_ = []  # 文字
            mention = []
            for key, value in self.data[item]['pair_candidates'].items():
                for v in value:  # each entity_relation pair
                    pair.append(''.join(v))
                    mention.append(key)
                    pair_.append(v)

            if not pair:
                pair.append('<None>+<None>')
                pair_.append(['<None>', '+', '<None>'])
                mention.append('None')

            input_ids = []
            token_type_ids = []
            for each in pair:
                enc = self.tokenizer.encode_plus(
                    question,
                    each,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
                input_ids.append(enc['input_ids'])
                token_type_ids.append(enc['token_type_ids'])
            if self.is_test:
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                    'pair_': pair_,  # 文字
                    'mention': mention,  # 后面用于对mention取topk
                    'origin_data': self.data[item]  # 为了下游任务
                }
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'pair_': pair_,  # 文字
                'mention': mention,  # 后面用于对mention取topk
                'true_pair': true_pair_,  # 文字
                'origin_data': self.data[item]  # 为了下游任务
            }

    def __len__(self):
        return len(self.data)


class QGDataset(Dataset):
    def __init__(self, data, device, is_train, is_test=False):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)
        self.max_len = 64

        self.data = data
        self.device = device
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        node = self.data[item]

        ques = node['question']

        if not self.is_test:
            qg = sum(list(node['qg_candidates'].values()), [])
            if len(qg) > 64:
                qg = random.sample(qg, 64)
            for each in qg:
                if len(each) == len(node['parse']) and all([e in each for e in node['parse']]):
                    qg.remove(each)
            label = [0] * len(qg)
            if len(node['parse']) <= 3:
                for each in list(permutations(node['parse'])):  # 将正确答案的path排列作为正例
                    if each:
                        qg.append(list(each))
                        label.append(1)
            else:
                qg.append(node['parse'])
                label.append(1)
        else:
            qg = sum(list(node['qg_candidates'].values()), [])

        qg_sentence = []
        for each in qg:
            ps = []
            for path in each:
                ps.append(''.join(path))
            qg_sentence.append(''.join(ps))  # list of string
        if not qg_sentence:
            qg_sentence.append('<None>+<None>')
            qg.append([['<None>', '+', '<None>']])

        input_ids = []
        token_type_ids = []
        for each in qg_sentence:
            enc = self.tokenizer.encode_plus(
                ques,
                each,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True
            )
            input_ids.append(enc['input_ids'])
            token_type_ids.append(enc['token_type_ids'])

        if self.is_train:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
            }
        else:
            if self.is_test:
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                    'qg_': qg,  # 文字
                    'origin_data': node
                }
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'qg_': qg,  # 文字
                'parse': node['parse'],  # 正确查询图
                'origin_data': node
            }


class NERDataset(Dataset):
    def __init__(self, data, device, is_train, is_test=False):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)
        self.max_len = 64

        self.data = data
        self.device = device
        self.is_train = is_train
        self.is_test = is_test

    def __getitem__(self, item):
        '''
        question: question
        tags: entity_candidates keys
        :param item:
        :return:
        '''
        question = self.data[item]['question']
        ques_enc = self.tokenizer.encode_plus(
            question,
            add_special_tokens=False,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = [ques_enc['input_ids']]
        token_type_ids = [ques_enc['token_type_ids']]

        if self.is_test:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'question': question,
                'origin_data': self.data[item]  # 为了下游任务
            }
        mentions = list(self.data[item]['entity_mention'].keys())
        # O S B I
        label = question
        for m in mentions:
            length = len(m)
            if length == 1:
                label = label.replace(m, '1')
            else:
                label = label.replace(m, '2' + '3' * (length - 1))
        label_temp = [0] * self.max_len
        for i in range(len(label)):
            if label[i] in '123':
                label_temp[i] = int(label[i])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
            'label': torch.tensor([label_temp], dtype=torch.long).to(device=self.device),
            'question': question,
            'mention': mentions
        }

    def __len__(self):
        return len(self.data)


class EntityDataset(Dataset):
    def __init__(self, data, device, is_train, is_test=False):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)
        self.max_len = 64

        self.data = data
        self.device = device
        self.is_train = is_train
        self.is_test = is_test

    def __getitem__(self, item):
        question = self.data[item]['question']

        mentions = []
        entity_candidates = []

        for mention, entities in self.data[item]['entity_candidates'].items():
            for e in entities:
                mentions.append(mention)
                entity_candidates.append(e)

        if not entity_candidates:
            mentions.append('<None>')
            entity_candidates.append('<None>')

        input_ids = []
        token_type_ids = []
        for each in entity_candidates:
            enc = self.tokenizer.encode_plus(
                question,
                each,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True
            )
            input_ids.append(enc['input_ids'])
            token_type_ids.append(enc['token_type_ids'])

        if self.is_test:
            return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                    'entity': entity_candidates,  # 文字
                    'mention': mentions,  # 后面用于对mention取topk
                    'origin_data': self.data[item]  # 为了下游任务
                }

        entity = [each[0] for each in self.data[item]['parse']]
        label = []
        for e in entity_candidates:
            if e in entity:
                label.append(1)
            else:
                label.append(0)
        if self.is_train:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                'entity': entity_candidates,  # 文字
                'mention': mentions,  # 后面用于对mention取topk
                'true_entity': entity,  # 正确的entity
                'origin_data': self.data[item]  # 为了下游任务
            }

    def __len__(self):
        return len(self.data)
