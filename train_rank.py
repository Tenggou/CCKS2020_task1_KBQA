import argparse
import ast
import json
import random
from datetime import datetime
from itertools import permutations

import numpy as np

import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

from models import Model
from train_filter import load_vocabulary
from train_rank_prepare import evaluate_qg_precision


class RankDataset(Dataset):
    def __init__(self, data, device, is_train, vocabulary, model_name, is_test=False):

        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('resources/bert-base-chinese-vocab.txt')
            self.max_len = 64
        elif model_name == 'lstm':
            self.vocabulary = vocabulary

        self.data = data
        self.model_name = model_name
        self.device = device
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        node = self.data[item]

        ques = node['question']

        if self.is_train:
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

        if self.model_name == 'bert':
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
        elif self.model_name == 'lstm':
            def get_index(string):
                return [self.vocabulary[q] if q in self.vocabulary else self.vocabulary['UNK'] for q in list(string)]

            ques = get_index(ques)
            qg_temp = []
            for each in qg_sentence:
                qg_temp.append(get_index(each))
            qg_sentence = qg_temp

        if self.is_train:
            if self.model_name == 'bert':
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device=self.device),
                    'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device=self.device),
                    'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
                }
            elif self.model_name == 'lstm':
                return {
                    'ques': [torch.tensor(ques, dtype=torch.long).to(device=self.device)] * len(qg),
                    'qg': [torch.tensor(each, dtype=torch.long).to(device=self.device) for each in qg_sentence],
                    'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
                }
        else:
            if self.model_name == 'bert':
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
            elif self.model_name == 'lstm':
                if self.is_test:
                    return {
                        'ques': [torch.tensor(ques, dtype=torch.long).to(device=self.device)] * len(qg),
                        'qg': [torch.tensor(each, dtype=torch.long).to(device=self.device) for each in qg_sentence],
                        'qg_': qg,  # 文字
                        'origin_data': node
                    }
                return {
                    'ques': [torch.tensor(ques, dtype=torch.long).to(device=self.device)] * len(qg),
                    'qg': [torch.tensor(each, dtype=torch.long).to(device=self.device) for each in qg_sentence],
                    'qg_': qg,  # 文字
                    'parse': node['parse'],  # 正确查询图
                    'origin_data': node
                }


def train_rank_collate(batch_data):
    if 'ques' in batch_data[0]:  # lstm
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'qg': [ent for data in batch_data for ent in data['qg']],
            'label': torch.cat(tuple([data['label'] for data in batch_data]))
        }
    elif 'input_ids' in batch_data[0]:  # bert
        return {
            'input_ids': torch.cat(tuple([data['input_ids'] for data in batch_data])),
            'token_type_ids': torch.cat(tuple([data['token_type_ids'] for data in batch_data])),
            'label': torch.cat(tuple([data['label'] for data in batch_data]))  # .half()  # 半精度
        }


def test_collate(batch_data):
    return batch_data[0]


def train_loop(model, loss_fn, optimizer, epochs, batch_size, model_path, train_data, dev_data, device, model_name,
               vocabulary):
    train_time = datetime.now()

    dev_data = DataLoader(
        RankDataset(data=dev_data, device=device, is_train=False, model_name=model_name, vocabulary=vocabulary),
        batch_size=1, num_workers=0, collate_fn=test_collate)
    train_data = DataLoader(
        RankDataset(data=train_data, device=device, is_train=True, model_name=model_name, vocabulary=vocabulary),
        batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=train_rank_collate)

    flag = 0
    all_loss = []
    result = evaluate(model=model, data=dev_data)
    # result = 0
    for epoch in range(epochs):
        print('\n\n%d epoch train start' % (int(epoch) + 1))
        epoch_time = datetime.now()
        epoch_loss = []

        for i, item in enumerate(train_data):
            epoch_loss.append(model.train_rank(item, loss_fn=loss_fn, optimizer=optimizer, i=i))

        epoch_loss = np.mean(epoch_loss)
        all_loss.append(epoch_loss)
        print('%d epoch train loss is ' % (int(epoch) + 1), epoch_loss, ', used time ', datetime.now() - epoch_time)

        eva_res = evaluate(model=model, data=dev_data)
        if eva_res > result:
            print('saving model ...')
            result = eva_res
            model.save(model_path=model_path, result=result)
            flag = 0
        else:
            flag += 1
        if flag >= 3:
            break

    print('\nbest accuracy is %.3f' % result)
    # print('all loss are', all_loss)
    print('train used time is', datetime.now() - train_time, '\n')


def evaluate(model, data):
    print('test start ...')
    start_time = datetime.now()
    count = 0
    length = 0
    for _, item in enumerate(data):
        length += 1
        index = model.predict_rank(item)
        pred = item['qg_'][index]
        if len(pred) == len(item['parse']) and all([e in pred for e in item['parse']]):
            count += 1
            if item['parse'] in error:
                print(pred)
                print(item['parse'])
    accuracy = count / length
    print('count: %d, length: %d' % (count, length))
    print('accuracy is %.3f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='gpu, epochs, batch_size, lr, is_train')
    parser.add_argument('--gpu', '-gpu', help='str :0, 1, 选择GPU， 默认0', default='0', type=str)
    parser.add_argument('--model_name', '-name', help='str', default='bert', type=str)
    parser.add_argument('--epochs', '-epochs', help='int: 300', default=300, type=int)
    parser.add_argument('--batch_size', '-batch', help='int', default=4, type=int)  # lstm:50 bert:4
    parser.add_argument('--lr', '-lr', help='学习率, 默认1e-3', default=2e-5, type=float)  # lstm: 1e-3 bert: 2e-5
    parser.add_argument('--is_train', '-train', help='bool, 默认True', default=True, type=ast.literal_eval)
    parser.add_argument('--model_path', '-path', help='模型路径, str', default='model/rank/', type=str)
    parser.add_argument('--is_load', '-load', help='bool, 默认False', default=False, type=ast.literal_eval)
    args = parser.parse_args()

    gpu = args.gpu
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    is_train = args.is_train
    model_path = args.model_path
    is_load = args.is_load

    model_path += model_name

    device = 'cuda: ' + gpu if torch.cuda.is_available() else 'cpu'

    # 速度提升不明显
    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
    print('device name ', device)

    vocabulary = load_vocabulary()
    model = Model(device=device, size_vocabulary=len(vocabulary) + 1, model_name=model_name)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(filter(lambda x: x.requires_grad, model.encoder.parameters())), lr=lr)

    dev_data = json.load(open('data/train_rank/dev.json', 'r', encoding='utf-8'))
    # 测试集上限 0.738
    error = evaluate_qg_precision(dev_data)

    if is_load or not is_train:
        model.load(model_path=model_path)

    if is_train:
        train_data = json.load(open('data/train_rank/train.json', 'r', encoding='utf-8'))
        train_loop(model, loss_fn, optimizer, epochs, batch_size, model_path, train_data, dev_data, device, model_name,
                   vocabulary)
    else:
        print('loading data ...')
        test_data = DataLoader(
            RankDataset(data=dev_data, device=device, is_train=False, model_name=model_name, vocabulary=vocabulary),
            batch_size=1, num_workers=0, collate_fn=test_collate)
        _ = evaluate(model=model, data=test_data)
    print('all used time is', datetime.now() - start_time)
    # lstm 56.25%
