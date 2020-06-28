import torch
import torch.nn as nn

from networks import BiLSTM, BERT
from datetime import datetime


class Model(object):
    def __init__(self, model_name, device, size_vocabulary, emb_dim=300, hidden_dim=300):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.model_name = model_name
        if model_name == 'lstm':
            self.encoder = BiLSTM(device=self.device, size_vocabulary=size_vocabulary).to(device=self.device)
        elif model_name == 'bert':
            self.encoder = BERT().to(device=self.device)
            # self.encoder.half()  # 半精度模型 float16， 容易出错

    def load(self, model_path):
        state_dict = torch.load(model_path + '.model', map_location=self.device)
        self.encoder.load_state_dict(state_dict['encoder'])
        self.encoder.train()
        print('loading model: ', model_path + '.model')
        print('result: ', state_dict['result'])
        print('saved time: ', state_dict['time'])

    def save(self, result, model_path):
        self.encoder.to('cpu')

        torch.save({
            'encoder': self.encoder.state_dict(),
            'result': result,
            'time': str(datetime.now())
        }, model_path + '.model')

        self.encoder.to(self.device)

    def train_filter(self, data, optimizer, loss_fn, clip=0.5, i=0):
        if self.model_name == 'lstm':
            optimizer.zero_grad()  # 清空优化器 why?
            ques = self.encoder(data['ques'])
            pair = self.encoder(data['pair'])
            scores = torch.sum(ques * pair, -1)

            loss = loss_fn(scores, data['label'])
            loss.backward()  # 反向传播, 计算梯度

            if self.model_name == 'lstm':
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()), clip)  # 防止梯度爆炸
            optimizer.step()  # 利用梯度，迭代参数

            return loss.item()
        elif self.model_name == 'bert':
            accumulation_steps = 10  # batch_size = 4, each backward = 40

            scores = self.encoder(data['input_ids'], data['token_type_ids'])
            loss = loss_fn(scores, data['label'])
            loss.backward()
            if (i + 1) % accumulation_steps == 0:  # 梯度累加
                optimizer.step()
                optimizer.zero_grad()
            return loss.item()

    def predict_filter(self, data, k):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            if self.model_name == 'lstm':
                ques = self.encoder(data['ques'])
                ent = self.encoder(data['pair'])
                scores = torch.sum(ques * ent, -1)
            elif self.model_name == 'bert':
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
            if len(scores) > k:
                _, predict_index = scores.topk(k)
            else:
                _, predict_index = scores.topk(len(scores))
            predict_index = predict_index.tolist()
            self.encoder.train()
            return predict_index

    def train_rank(self, data, optimizer, loss_fn, clip=0.5, i=0):
        if self.model_name == 'lstm':
            optimizer.zero_grad()  # 清空优化器 why?
            ques = self.encoder(data['ques'])
            pair = self.encoder(data['qg'])
            scores = torch.sum(ques * pair, -1)

            loss = loss_fn(scores, data['label'])
            loss.backward()  # 反向传播, 计算梯度
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()), clip)  # 防止梯度爆炸
            optimizer.step()  # 利用梯度，迭代参数
            return loss.item()
        elif self.model_name == 'bert':
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
            if self.model_name == 'lstm':
                ques = self.encoder(data['ques'])
                ent = self.encoder(data['qg'])
                scores = torch.sum(ques * ent, -1)
            elif self.model_name == 'bert':
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
