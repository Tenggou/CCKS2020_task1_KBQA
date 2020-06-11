import torch
import torch.nn as nn

from networks import BiLSTM
from datetime import datetime


class Filter(object):
    def __init__(self, emb_dim=300, hidden_dim=300, device=torch.device('cpu'), size_vocabulary=0):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = BiLSTM(device=self.device, size_vocabulary=size_vocabulary).to(device)
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def load(self, model_path='model/filter/'):
        state_dict = torch.load(model_path + '.model')
        self.encoder.load_state_dict(state_dict['encoder'])
        print('loading model: ', model_path+'.model')
        print('result: ', state_dict['result'])
        print('saved time: ', state_dict['time'])

    def save(self, result=0, model_path='model/filter/'):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'result': result,
            'time': str(datetime.now())
        }, model_path + '.model')

    def train(self, data, optimizer, loss_fn, clip=0.5):
        # start_time = datetime.now()

        optimizer.zero_grad()  # 清空优化器 why?
        ques = self.encoder(data['ques'])
        pair = self.encoder(data['pair'])
        # scores = self.cos(ques, ent)
        scores = torch.sum(ques * pair, -1)

        # print('forward time', datetime.now() - start_time)
        # start_time = datetime.now()

        label = torch.Tensor(data['label']).to(device=self.device)
        loss = loss_fn(scores, label)
        loss.backward()  # 反向传播, 计算梯度

        # print('backward time', datetime.now() - start_time)
        # start_time = datetime.now()

        nn.utils.clip_grad_norm_(list(self.encoder.parameters()), clip)  # 防止梯度爆炸
        optimizer.step()  # 利用梯度，迭代参数

        # print('optimize time', datetime.now() - start_time, '\n')

        return loss.item()

    def predict(self, data):
        with torch.no_grad():  # 无法计算梯度，能够节省算力
            self.encoder.eval()  # 影响一些功能，如dropout，batchnorm
            ques = self.encoder(data['ques'])
            ent = self.encoder(data['pair'])
            self.encoder.train()
            # scores = self.cos(ques, ent)
            scores = torch.sum(ques*ent, -1)

        _, predict_index = scores.topk(len(scores))
        return predict_index
