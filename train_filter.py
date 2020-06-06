import argparse
import ast
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim

import numpy as np
from torch.utils.data import DataLoader, Dataset

from models import Filter


class TrainFilterDataset(Dataset):
    def __init__(self, path, vocabulary):
        self.data = json.load(open(path, 'r', encoding='utf-8'))
        self.vocabulary = vocabulary

    def __getitem__(self, item):
        question = [[self.vocabulary[q] if q in self.vocabulary else self.vocabulary['UNK'] for q in
                     list(self.data[item]['question'])]]

        # 取出正确的实体(训练正例)
        true_entity = []  # index
        true_entity_ = []  # 文字
        for p in self.data[item]['parse']:
            true_entity.append(
                [self.vocabulary[t] if t in self.vocabulary else self.vocabulary['UNK'] for t in p['entity']])
            true_entity_.append(p['entity'])

        # 训练负例
        entity = []
        for key, value in self.data[item]['candidate_entity'].items():
            for v in value:  # each entity candidate
                if v not in true_entity_:
                    entity.append([self.vocabulary[e] if e in self.vocabulary else self.vocabulary['UNK'] for e in v])

        label = [0] * len(entity) + [1] * len(true_entity)
        entity += true_entity
        return {
            'ques': question * len(entity),
            'ent': entity,
            'label': label
        }

    def __len__(self):
        return len(self.data)


class TestFilterDataset(Dataset):
    def __init__(self, path, vocabulary):
        self.data = json.load(open(path, 'r', encoding='utf-8'))
        self.vocabulary = vocabulary

    def __getitem__(self, item):
        question = [[self.vocabulary[q] if q in self.vocabulary else self.vocabulary['UNK'] for q in
                     list(self.data[item]['question'])]]

        true_entity = []  # index
        for p in self.data[item]['parse']:
            true_entity.append(
                [self.vocabulary[t] if t in self.vocabulary else self.vocabulary['UNK'] for t in p['entity']])

        entity = []
        mention = []
        for key, value in self.data[item]['candidate_entity'].items():
            for v in value:  # each entity candidate
                entity.append([self.vocabulary[e] if e in self.vocabulary else self.vocabulary['UNK'] for e in v])
                mention.append(key)

        return {
            'ques': question,
            'ent': entity,
            'mention': mention,
            'true_ent': true_entity
        }

    def __len__(self):
        return len(self.data)


def test_filter_collate(batch_data):
    return batch_data[0]


def train_filter_collate(batch_data):
    # 将数据整理为能够并行处理的格式
    return {
        'ques': [ques for data in batch_data for ques in data['ques']],
        'ent': [ent for data in batch_data for ent in data['ent']],
        'label': [label for data in batch_data for label in data['label']]
    }


def train_loop(model, optimizer, loss_fn, model_path, batch_size, vocabulary):
    train_time = datetime.now()

    test_data = DataLoader(TestFilterDataset('data/train_filter/dev.json', vocabulary),
                           batch_size=1, num_workers=4,
                           collate_fn=test_filter_collate)#, pin_memory=True)  # ques, ent, true_ent

    train_data = DataLoader(TrainFilterDataset('data/train_filter/train.json', vocabulary),
                            batch_size=batch_size, num_workers=4,
                            collate_fn=train_filter_collate, shuffle=True)#, pin_memory=True)  # ques, ent, label

    best_res = 0
    flag = 0
    all_loss = []
    for epoch in range(epochs):
        print('\n\n%d epoch train start' % (int(epoch) + 1))
        epoch_time = datetime.now()
        epoch_loss = []

        # loaddatatime = datetime.now()
        for i, item in enumerate(train_data):
            # print('loaddatatime: ', datetime.now()-loaddatatime)
            epoch_loss.append(model.train(item, optimizer, loss_fn))
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


def evaluate(model, data):
    print('evaluating ...')
    start_time = datetime.now()
    accuracy = 0
    length = 0

    def calulate_acc(true, pred):
        acc = 0
        for key, value in pred.items():
            for v in value:
                if v in true:
                    acc += 1
        return acc / len(true)

    def get_topk_candidate(index, data, k):
        # get best candidate for each mention
        entity = {}
        for i in index:
            if data['mention'][i] not in entity and len(entity) < 3:
                entity[data['mention'][i]] = [data['ent'][i]]
            elif data['mention'][i] in entity and len(entity[data['mention'][i]]) < k:
                entity[data['mention'][i]].append(data['ent'][i])
        return entity

    for i, item in enumerate(data):
        index = model.predict(item).tolist()
        acc = calulate_acc(item['true_ent'], get_topk_candidate(index, item, k))
        accuracy += acc
        length += 1
    accuracy /= length
    print('accuracy is %.3f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='gpu, epochs, batch_size, lr, is_train')
    parser.add_argument('--gpu', '-gpu', help='str :0, 1, 选择GPU， 默认0', default='0', type=str)
    parser.add_argument('--epochs', '-epochs', help='int: 300', default=300, type=int)
    parser.add_argument('--batch_size', '-batch', help='int: 50', default=50, type=int)
    parser.add_argument('--lr', '-lr', help='学习率, 默认1e-3', default=1e-3, type=float)
    parser.add_argument('--is_train', '-train', help='bool, 默认True', default=True, type=ast.literal_eval)
    parser.add_argument('--model_path', '-path', help='模型路径, str', default='model/filter-', type=str)
    parser.add_argument('--k', '-k', help='int', default=3, type=int)
    args = parser.parse_args()

    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    is_train = args.is_train
    model_path = args.model_path
    k = args.k

    model_path += str(k)

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')  # 检测GPU
    # 提升不明显
    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
    print('device name ', device)

    # 加载字符
    def load_vocabulary():
        characters = json.load(open('resources/characters.txt', 'r', encoding='utf-8'))
        vocabulary = {}
        count = 1
        for c in characters:
            vocabulary[c] = count
            count += 1
        return vocabulary

    vocabulary = load_vocabulary()
    model = Filter(device=device, size_vocabulary=len(vocabulary) + 1)  # +1 为pad值，index=0

    # label 1/0   pointwise
    # loss_fn = nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder.parameters())), lr=lr)

    if is_train:
        train_loop(model=model, optimizer=optimizer, loss_fn=loss_fn, model_path=model_path,
                   batch_size=batch_size, vocabulary=vocabulary)
    else:
        print('load model ...')
        model.load(model_path)
        print('loading data ...')
        test_data = DataLoader(TestFilterDataset('data/train_filter/dev.json', vocabulary),
                               batch_size=1, num_workers=4, collate_fn=test_filter_collate)#, pin_memory=True)
        print('test start ...')
        _ = evaluate(model=model, data=test_data)

    print('all used time is', datetime.now() - start_time)
    # k=1 0.820 k=3 0.896 k=5 0.917 k=10 0.929
