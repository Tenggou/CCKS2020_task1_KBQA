import argparse
import ast
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch import optim

import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from models import Model
from train_filter_prepare import evaluate_pair_precision


class FilterDataset(Dataset):
    def __init__(self, data, vocabulary, device, is_train, model_name, is_test=False):
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

            # 每个就选取100个负例来训练
            if len(fake_pair) > 100:
                fake_pair = random.sample(fake_pair, 100)

            label = [0] * len(fake_pair) + [1] * len(true_pair)
            if self.model_name == 'lstm':
                question = self.get_index(question)
                pair = [self.get_index(each) for each in fake_pair + true_pair]
                return {
                    'ques': [torch.tensor(question).to(device=self.device)] * len(pair),
                    'pair': [torch.tensor(p).to(device=self.device) for p in pair],
                    'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
                }
            elif self.model_name == 'bert':
                ques = self.tokenizer.encode_plus(
                    question,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )['input_ids']

                pair = [self.tokenizer.encode_plus(
                    each,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )['input_ids'] for each in fake_pair + true_pair]
                return {
                    'ques': torch.tensor([ques] * len(pair), dtype=torch.long).to(device=self.device),
                    'pair': torch.tensor(pair, dtype=torch.long).to(device=self.device),
                    'label': torch.tensor(label, dtype=torch.float).to(device=self.device)
                }
        else:
            pair = []  # index
            pair_ = []  # 文字
            mention = []
            for key, value in self.data[item]['pair_candidates'].items():
                for v in value:  # each entity_relation pair
                    pair.append(''.join(v))
                    mention.append(key)
                    pair_.append(v)
            if self.model_name == 'lstm':
                question = self.get_index(question)
                pair = [self.get_index(each) for each in pair]

                if self.is_test:
                    return {
                        'ques': [torch.tensor(question).to(device=self.device)],
                        'pair': [torch.tensor(p).to(device=self.device) for p in pair],  # index
                        'pair_': pair_,  # 文字
                        'mention': mention,  # 后面用于对mention取topk
                        'origin_data': self.data[item]  # 为了下游任务
                    }
                return {
                    'ques': [torch.tensor(question).to(device=self.device)],
                    'pair': [torch.tensor(p).to(device=self.device) for p in pair],  # index
                    'pair_': pair_,  # 文字
                    'mention': mention,  # 后面用于对mention取topk
                    'true_pair': true_pair_,  # 文字
                    'origin_data': self.data[item]  # 为了下游任务
                }
            elif self.model_name == 'bert':
                ques = self.tokenizer.encode_plus(
                    question,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )['input_ids']
                pair = [self.tokenizer.encode_plus(
                    each,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )['input_ids'] for each in pair]
                if self.is_test:
                    return {
                        'ques': torch.tensor([ques] * len(pair), dtype=torch.long).to(device=self.device),
                        'pair': torch.tensor(pair, dtype=torch.long).to(device=self.device),
                        'pair_': pair_,  # 文字
                        'mention': mention,  # 后面用于对mention取topk
                        'origin_data': self.data[item]  # 为了下游任务
                    }
                return {
                    'ques': torch.tensor([ques] * len(pair), dtype=torch.long).to(device=self.device),
                    'pair': torch.tensor(pair, dtype=torch.long).to(device=self.device),
                    'pair_': pair_,  # 文字
                    'mention': mention,  # 后面用于对mention取topk
                    'true_pair': true_pair_,  # 文字
                    'origin_data': self.data[item]  # 为了下游任务
                }

    def __len__(self):
        return len(self.data)

    def get_index(self, string):
        return [self.vocabulary[q] if q in self.vocabulary else self.vocabulary['UNK'] for q in list(string)]


def test_collate(batch_data):
    return batch_data[0]


def train_filter_collate(batch_data):
    # 将数据整理为能够并行处理的格式
    if type(batch_data[0]['ques']) == list:
        return {
            'ques': [ques for data in batch_data for ques in data['ques']],
            'pair': [ent for data in batch_data for ent in data['pair']],
            'label': torch.cat(tuple([data['label'] for data in batch_data]))
        }
    elif type(batch_data[0]['ques']) == torch.Tensor:
        return {
            'ques': torch.cat(tuple([data['ques'] for data in batch_data])),
            'pair': torch.cat(tuple([data['pair'] for data in batch_data])),
            'label': torch.cat(tuple([data['label'] for data in batch_data]))
        }


def train_loop(model, loss_fn, optimizer, epochs, batch_size, model_path, train_data, dev_data, device, model_name,
                   vocabulary):
    train_time = datetime.now()

    test_data = DataLoader(FilterDataset(dev_data, vocabulary, device, False, model_name), batch_size=1, num_workers=0, collate_fn=test_collate)

    train_data = DataLoader(FilterDataset(train_data, vocabulary, device, True, model_name), batch_size=batch_size, num_workers=0, collate_fn=train_filter_collate, shuffle=True)

    best_res = evaluate(model=model, data=test_data)
    flag = 0
    all_loss = []
    for epoch in range(epochs):
        print('\n\n%d epoch train start' % (int(epoch) + 1))
        epoch_time = datetime.now()
        epoch_loss = []

        # loaddatatime = datetime.now()
        for i, item in enumerate(train_data):
            # print('loaddatatime: ', datetime.now()-loaddatatime)
            epoch_loss.append(model.train_filter(item, optimizer, loss_fn))
            # loaddatatime = datetime.now()

        epoch_loss = np.mean(epoch_loss)
        all_loss.append(epoch_loss)
        print('%d epoch train loss is ' % (int(epoch) + 1), epoch_loss,
              ', used time ', datetime.now() - epoch_time)

        print('test start ...')
        eva_res = evaluate(model=model, data=test_data)
        if eva_res > best_res:
            print('saving model ...')
            best_res = eva_res
            model.save(model_path=model_path, result=best_res)
            flag = 0
        else:
            flag += 1
        if flag >= 5:
            break
    print('\nbest accuracy is %.3f' % best_res)
    # print('all loss are', all_loss)
    print('train used time is', datetime.now() - train_time, '\n')


def get_topk_candidate(index, data):
    # get topk candidates and sort into mentions
    pair = {}
    for i in index:
        # if data['mention'][i] not in pair and len(pair) < 3:
        #     pair[data['mention'][i]] = [data['pair_'][i]]
        # elif data['mention'][i] in pair and len(pair[data['mention'][i]]) < k:
        #     pair[data['mention'][i]].append(data['pair_'][i])
        if any([char in data['pair_'][i][0][1:-1] or char in data['pair_'][i][-1][1:-1] for char in
                [' ', '|', '<', '>', '"', '{', '}', '\\']]):  # sparql query报错了
            continue
        if data['mention'][i] not in pair:
            pair[data['mention'][i]] = [data['pair_'][i]]
        elif data['mention'][i] in pair:
            pair[data['mention'][i]].append(data['pair_'][i])
    return pair


def evaluate(model, data):
    print('evaluating ...')
    start_time = datetime.now()
    accuracy = 0
    length = 0

    def calulate_acc(true, pred):
        if len(true) == 0:
            return 0
        acc = 0
        for v in sum(list(pred.values()), []):
            if v in true:
                acc += 1
        return acc / len(true)

    for i, item in enumerate(data):
        index = model.predict_filter(item, k)
        acc = calulate_acc(item['true_pair'], get_topk_candidate(index, item))
        accuracy += acc
        length += 1
    accuracy /= length
    print('accuracy is %.3f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


def load_vocabulary():
    # 加载字符
    characters = json.load(open('resources/characters.txt', 'r', encoding='utf-8'))
    if '+' not in characters:
        characters.append('+')
    if '-' not in characters:
        characters.append('-')
    vocabulary = {}
    count = 1
    for c in characters:
        vocabulary[c] = count
        count += 1
    return vocabulary


if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='gpu, epochs, batch_size, lr, is_train')
    parser.add_argument('--gpu', '-gpu', help='str :0, 1, 选择GPU， 默认0', default='0', type=str)
    parser.add_argument('--model_name', '-name', help='str', default='lstm', type=str)
    parser.add_argument('--epochs', '-epochs', help='int: 300', default=300, type=int)
    parser.add_argument('--batch_size', '-batch', help='int: 50', default=50, type=int)
    parser.add_argument('--lr', '-lr', help='学习率, 默认1e-3', default=1e-3, type=float)
    parser.add_argument('--is_train', '-train', help='bool, 默认True', default=True, type=ast.literal_eval)
    parser.add_argument('--is_load', '-load', help='bool, False', default=False, type=ast.literal_eval)
    parser.add_argument('--model_path', '-path', help='模型路径, str', default='model/filter/', type=str)
    parser.add_argument('--k', '-k', help='int', default=3, type=int)
    args = parser.parse_args()

    gpu = args.gpu
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    is_train = args.is_train
    is_load = args.is_load
    model_path = args.model_path
    k = args.k

    model_path += model_name+str(k)

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')  # 检测GPU
    # 速度提升不明显
    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
    print('device name ', device)

    vocabulary = load_vocabulary()
    model = Model(device=device, size_vocabulary=len(vocabulary)+1, model_name=model_name)  # +1 为pad值，index=0

    if is_load or not is_train:
        model.load(model_path)

    # label 1/0   pointwise
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder.parameters())), lr=lr)

    dev_data = json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    train_data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8'))
    # 测试集的准确度上限 0.9079
    evaluate_pair_precision(dev_data)

    if is_train:
        train_loop(model, loss_fn, optimizer, epochs, batch_size, model_path, train_data, dev_data, device, model_name,
                   vocabulary)
    else:
        print('loading data ...')
        test_data = DataLoader(FilterDataset(dev_data, vocabulary, device, False, model_name), batch_size=1, num_workers=0, collate_fn=test_collate)
        print('test start ...')
        evaluate(model=model, data=test_data)

    print('all used time is', datetime.now() - start_time)
    # BILSTM: 10: 0.833
