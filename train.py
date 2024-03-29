import argparse
import ast
import json
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model.model import Model
from utils.configure import K_PAIR, FILTER_DEV_DATA_PATH, FILTER_TRAIN_DATA_PATH, RANK_DEV_DATA_PATH, \
    RANK_TRAIN_DATA_PATH
from utils.dataset import FilterDataset, RankDataset, test_collate, train_collate
from utils.evaluate import evaluate_filter, evaluate_rank


def train_loop():
    train_time = datetime.now()

    best_res = evaluate(model=model, data=dev_data)
    flag = 0
    all_loss = []
    for epoch in range(epochs):
        print('\n\n%d epoch train start' % (int(epoch) + 1))
        epoch_time = datetime.now()
        epoch_loss = []

        for i, item in enumerate(train_data):
            epoch_loss.append(model.train(item, optimizer, loss_fn, i=i))

        epoch_loss = np.mean(epoch_loss)
        all_loss.append(epoch_loss)
        print('%d epoch train loss is ' % (int(epoch) + 1), epoch_loss,
              ', used time ', datetime.now() - epoch_time)

        print('test start ...')
        eva_res = evaluate(model=model, data=dev_data)
        if eva_res > best_res:
            print('saving model ...')
            best_res = eva_res
            model.save(result=best_res)
            flag = 0
        else:
            flag += 1
        if flag >= 3:
            break
    print('\nbest accuracy is %.4f' % best_res)
    # print('all loss are', all_loss)
    print('train used time is', datetime.now() - train_time, '\n')


if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='gpu, epochs, batch_size, lr, is_train')
    parser.add_argument('--gpu', '-gpu', help='选择GPU， 默认0', default='0', type=str)
    parser.add_argument('--epochs', '-epochs', help='最大训练次数，默认100', default=100, type=int)
    parser.add_argument('--batch_size', '-batch', help='一批数据的个数，默认4', default=4, type=int)
    parser.add_argument('--lr', '-lr', help='学习率, 默认1e-3', default=2e-5, type=float)
    parser.add_argument('--is_train', '-train', help='训练/测试, True/False，默认True', default=True, type=ast.literal_eval)
    parser.add_argument('--is_load', '-load', help='是否加载模型参数， 默认False', default=False, type=ast.literal_eval)
    parser.add_argument('--component', '-component', help='模块，默认为filter', default='filter', type=str)
    args = parser.parse_args()

    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    is_train = args.is_train
    is_load = args.is_load
    component = args.component

    # 模型参数地址
    parameter_path = 'parameter/' + component
    parameter_path += '.' + str(K_PAIR) + '.model' if component == 'filter' else '.model'
    print(f'parameter path is {parameter_path}')

    # 选择GPU
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():  # 速度提升不明显
        from torch.backends import cudnn

        cudnn.benchmark = True
    print('device name is ', device)

    # 初始化模型
    model = Model(device=device, component=component, model_path=parameter_path)
    if is_load or not is_train:
        print('loading model parameter ... ')
        model.load()

    # label 1/0   pointwise
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder.parameters())), lr=lr)

    # 加载数据
    if component == 'filter':
        dev_data_path = FILTER_DEV_DATA_PATH
        train_data_path = FILTER_TRAIN_DATA_PATH
        Dataset = FilterDataset

        evaluate = evaluate_filter
    else:
        dev_data_path = RANK_DEV_DATA_PATH
        train_data_path = RANK_TRAIN_DATA_PATH
        Dataset = RankDataset

        evaluate = evaluate_rank

    print('loading data .... ')
    dev_data = json.load(open(dev_data_path, 'r', encoding='utf-8'))
    train_data = json.load(open(train_data_path, 'r', encoding='utf-8'))
    dev_data = DataLoader(Dataset(data=dev_data, device=device, is_test=False, is_train=False),
                          batch_size=1, num_workers=0, collate_fn=test_collate)
    train_data = DataLoader(Dataset(data=train_data, device=device, is_test=False, is_train=is_train),
                            batch_size=batch_size, num_workers=0, collate_fn=train_collate, shuffle=True)

    # 训练/测试
    if is_train:
        train_loop()
    else:
        print('test start ...')
        evaluate(model=model, data=dev_data)

    print('all used time is', datetime.now() - start_time)
