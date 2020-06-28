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


def get_error_result():
    data = json.load(open('data/our_dev_result.json', 'r', encoding='utf-8'))
    error = []
    for node in data:
        for answer in node['answer']:
            if answer not in node['pred_answer']:
                if node not in error:
                    error.append(node)
                    break
        for answer in node['pred_answer']:
            if answer not in node['answer']:
                if node not in error:
                    error.append(node)
                    break
    print('%d data predict wrong answers. ' % len(error))
    error = random.sample(error, 100) if len(error) > 100 else error
    print('%d data predict wrong answers, using for error analyis. ' % len(error))
    json.dump(error, open('data/error_analysis.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


def error_analysis():
    error = json.load(open('data/error_analysis.json', 'r', encoding='utf-8'))
    error_dict = {
        'answer error': 0,
        'query graph error': 0,
        'entity-relation pair error': 0,
        'entity number error': 0
    }

    for node in error:
        if len(node['pred']) == len(node['parse']) and all([e in node['pred'] for e in node['parse']]):
            error_dict['answer error'] += 1
            # print(node['answer'])
            # print(node['pred_answer'])
            # print(node['parse'])
            # print(node['pred'])
        else:
            if len(node['pred']) != len(node['parse']):
                error_dict['entity number error'] += 1
                # print(node['parse'])
                # print(node['pred'])
            elif not all([any([parse[:3] == pred[:3] for parse in node['parse']]) for pred in node['pred']]):
                error_dict['entity-relation pair error'] += 1
                # print(node['parse'])
                # print(node['pred'])
                # print([any([parse[:3] == pred[:3] for parse in node['parse']]) for pred in node['pred']])
            elif not all([any([parse[:3] == pred[:3] for pred in node['pred']]) for parse in node['parse']]):
                error_dict['entity-relation pair error'] += 1
                # print(node['parse'])
                # print(node['pred'])
                # print([any([parse[:3] == pred[:3] for parse in node['parse']]) for pred in node['pred']])
            else:
                error_dict['query graph error'] += 1
                print(node['parse'])
                print(node['pred'])
    print(error_dict)


if __name__ == '__main__':

    # check_query_grap()

    # get_error_result()

    error_analysis()