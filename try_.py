import json
from datetime import datetime

import torch

from networks import BiLSTM

if __name__ == '__main__':

    # # !todo 查询图形状
    # templates = {}
    # data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8')) + \
    #        json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    # for node in data:
    #     template = []
    #     for p in node['parse']:
    #         template.append(p['direction'])
    #     template = '_'.join(template)
    #     if template in templates:
    #         templates[template] += 1
    #     else:
    #         templates[template] = 1
    # print(json.dumps(templates, indent=4))

    # test BiLSTM
    # a = torch.randn(10, 10)
    # b = torch.randn(10, 10)
    # c = torch.Tensor([6])
    # d = torch.Tensor([7, 8, 9, 10])
    # batch = [a, b, c, d]
    # model = BiLSTM(size_vocabulary=20)
    # model(batch)
    # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    # print(cos(a, b))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    import torchvision.models as models
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    dataset = datasets.FakeData(
        size=1000,
        transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        num_workers=1,
        pin_memory=True
    )

    model.to('cuda: 1')

    for data, target in loader:
        data = data.to('cuda: 1', non_blocking=True)
        target = target.to('cuda: 1', non_blocking=True).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Done')