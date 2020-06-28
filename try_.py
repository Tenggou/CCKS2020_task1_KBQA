import json
import random
import re
from datetime import datetime

import torch
from SPARQLWrapper import SPARQLWrapper, JSON


def check_query_grap():
    # 查询图形状
    templates = {}
    data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8')) + \
           json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    print(len(data))
    for node in data:
        template = []
        for p in node['parse']:
            t = ''
            for e in p:
                if e == '+' or e == '-':
                    t += e
            template.append(t)
        template = sorted(template, key=lambda x: len(x))
        template = '_'.join(template)
        if template in templates:
            templates[template] += 1
        else:
            templates[template] = 1
    templates = sorted(templates.items(), key=lambda x: x[1])
    print(json.dumps(templates, indent=4))


if __name__ == '__main__':
    # check_query_grap()

    data = json.load(open('data/our_dev_result.json', 'r', encoding='utf-8'))
    json.dump(random.sample(data, 100), open('data/error_analysis.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
