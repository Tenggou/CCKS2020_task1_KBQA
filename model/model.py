import torch
import torch.nn as nn

from model.transformers import BertModel
from datetime import datetime

from utils.configure import BERT_BASE_CHINESE, K_PAIR


class Model(object):
    def __init__(self, device, component, model_path):
        self.device = device
        self.encoder = BERT().to(device=self.device)
        # self.encoder.half()  # 半精度模型 float16， 容易出错
        self.train = self.train_filter if component == 'filter' else self.train_rank
        self.predict = self.predict_filter if component == 'filter' else self.predict_rank

        self.model_path = model_path

    def load(self):
        state_dict = torch.load(self.model_path + '.model', map_location=self.device)  # 加载数据到指定设备
        self.encoder.load_state_dict(state_dict['encoder'])
        self.encoder.train()
        print('loading model: ', self.model_path + '.model')
        print('result: ', state_dict['result'])
        print('saved time: ', state_dict['time'])

    def save(self, result):
        self.encoder.to('cpu')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'result': result,
            'time': str(datetime.now())
        }, self.model_path + '.model')
        self.encoder.to(self.device)

    def train_filter(self, data, optimizer, loss_fn, i=0):
        accumulation_steps = 10  # 梯度累加 batch_size = 4, each backward = 40
        scores = self.encoder(data['input_ids'], data['token_type_ids'])
        loss = loss_fn(scores, data['label'])
        loss.backward()
        if (i + 1) % accumulation_steps == 0:  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def predict_filter(self, data):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            if data['input_ids'].size()[0] <= 1500:
                scores = self.encoder(data['input_ids'], data['token_type_ids'])  # 容易爆显存
            else:
                scores = []
                for i in range(1, int(data['input_ids'].size()[0] / 1500 + 1 + 1)):  # +1向上取整 +1 range范围
                    if data['input_ids'][1500 * (i - 1):1500 * i].size()[0] == 0:
                        continue
                    scores += self.encoder(data['input_ids'][1500 * (i - 1):1500 * i],
                                           data['token_type_ids'][1500 * (i - 1):1500 * i]).tolist()
                scores = torch.tensor(scores)
            if len(scores) > K_PAIR:
                _, predict_index = scores.topk(K_PAIR)
            else:
                _, predict_index = scores.topk(len(scores))
            predict_index = predict_index.tolist()
            self.encoder.train()
            return predict_index

    def train_rank(self, data, optimizer, loss_fn, i=0):
        accumulation_steps = 10
        scores = self.encoder(data['input_ids'], data['token_type_ids'])
        loss = loss_fn(scores, data['label'])
        loss.backward()
        if (i + 1) % accumulation_steps == 0:  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def predict_rank(self, data):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            if data['input_ids'].size()[0] <= 1500:
                scores = self.encoder(data['input_ids'], data['token_type_ids'])  # 容易爆显存
            else:
                scores = []
                for i in range(1, int(data['input_ids'].size()[0] / 1500 + 1 + 1)):  # +1向上取整 +1 range范围
                    if data['input_ids'][1500 * (i - 1):1500 * i].size()[0] == 0:
                        continue
                    scores += self.encoder(data['input_ids'][1500 * (i - 1):1500 * i],
                                           data['token_type_ids'][1500 * (i - 1):1500 * i]).tolist()
                scores = torch.tensor(scores)
            predict_index = scores.argmax().item()
            self.encoder.train()
            return predict_index


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertModel.from_pretrained(BERT_BASE_CHINESE)
        self.encoder.train()
        self.output_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids):
        return self.output_layer(self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)[1]).view(-1)
