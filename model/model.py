import torch
import torch.nn as nn

from model.CRF import CRF
from model.transformers import BertModel
from datetime import datetime

from utils.configure import BERT_BASE_CHINESE


class Model(object):
    def __init__(self, device, component, model_path_load, model_path_save, k=5):
        self.device = device
        if component != 'ner':
            self.encoder = BERT().to(device=self.device)
            if component == 'entity' or component == 'pair':
                self.train = self.train_entity_and_pair
                self.predict = self.predict_entity_and_pair
            elif component == 'qg':
                self.train =  self.train_qg
                self.predict = self.predict_qg
        else:
            self.encoder = BERT_CRF().to(device=self.device)
            self.train = self.train_ner
            self.predict = self.predict_ner
        self.component = component
        self.model_path_load = model_path_load
        self.model_path_save = model_path_save
        self.k = k

    def load(self):
        state_dict = torch.load(self.model_path_load, map_location=self.device)  # 加载数据到指定设备
        self.encoder.load_state_dict(state_dict['encoder'])
        self.encoder.train()
        print('loading model: ', self.model_path_load)
        print('result: ', state_dict['result'])
        print('saved time: ', state_dict['time'])

    def save(self, result):
        self.encoder.to('cpu')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'result': result,
            'time': str(datetime.now())
        }, self.model_path_save)
        self.encoder.to(self.device)

    def train_entity_and_pair(self, data, optimizer, loss_fn, i=0):
        accumulation_steps = 10  # 梯度累加 batch_size = 4, each backward = 40
        scores = self.encoder(data['input_ids'], data['token_type_ids'])
        loss = loss_fn(scores, data['label'])
        loss.backward()
        if (i + 1) % accumulation_steps == 0:  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def predict_entity_and_pair(self, data):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            if data['input_ids'].size()[0] <= 1500:
                scores = self.encoder(data['input_ids'], data['token_type_ids'])  # 容易爆显存
            else:
                scores = []
                for i in range(1, int(data['input_ids'].size()[0] / 1500 + 1 + 1)):  # +1 向上取整 +1 range范围
                    if data['input_ids'][1500 * (i - 1):1500 * i].size()[0] == 0:
                        continue
                    scores += self.encoder(data['input_ids'][1500 * (i - 1):1500 * i],
                                           data['token_type_ids'][1500 * (i - 1):1500 * i]).tolist()
                scores = torch.tensor(scores)
            if len(scores) > self.k:
                _, predict_index = scores.topk(self.k)
            else:
                _, predict_index = scores.topk(len(scores))
            predict_index = predict_index.tolist()
            self.encoder.train()
            return predict_index

    def train_qg(self, data, optimizer, loss_fn, i=0):
        accumulation_steps = 10
        scores = self.encoder(data['input_ids'], data['token_type_ids'])
        loss = loss_fn(scores, data['label'])
        loss.backward()
        if (i + 1) % accumulation_steps == 0:  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def predict_qg(self, data):
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

    def train_ner(self, data, optimizer, loss_fn, i=0):
        scores = self.encoder(data['input_ids'], data['token_type_ids'], data['label'])
        scores = torch.abs(scores)
        scores.backward()
        optimizer.step()
        optimizer.zero_grad()
        return scores.item()

    def predict_ner(self, data):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            tags = self.encoder.predict(data['input_ids'], data['token_type_ids'])  # 容易爆显存
            self.encoder.train()
            return tags


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertModel.from_pretrained(BERT_BASE_CHINESE)
        self.encoder.train()
        self.output_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids):
        return self.output_layer(self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)[1]).view(-1)


class BERT_CRF(nn.Module):
    def __init__(self):
        super(BERT_CRF, self).__init__()
        self.encoder = BertModel.from_pretrained(BERT_BASE_CHINESE)
        self.encoder.train()
        # self.lstm = nn.LSTM(input_size=768,
        #                    hidden_size=4,
        #                    num_layers=1,
        #                    bidirectional=True,  #
        #                    batch_first=True)
        self.crf = CRF(num_tags=4, batch_first=True)  # O S B I
        self.trans_layer = nn.Linear(768, 4)

    def forward(self, input_ids, token_type_ids, tags):
        return self.crf(self.trans_layer(self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)[0]), tags)

    def predict(self, input_ids, token_type_ids):
        return self.crf.decode(self.trans_layer(self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)[0]))
