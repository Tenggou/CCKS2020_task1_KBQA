import sys
import os

sys.path.append(os.path.abspath(''))

from utils.evaluate import evaluate_pair_precision, evaluate_entity_precision

import json
import random
import re

import jieba
import pkuseg
from datetime import datetime

from utils.configure import STOPWORDS_PATH, TRAIN_DATA_PATH, PKUBASE_MENTION2ENT_COLLECTED, \
    FILTER_TRAIN_DATA_PATH, FILTER_DEV_DATA_PATH, OUR_JIEBA_DICT, QUERY_GRAPH
from utils.sparql import sparql_query, sparql_parser


def train_filter_prepare(data_path):
    '''
    20200516
    1. sparql_parser 对原数据进行解析，获得问题、实体、实体到答案的路径、分词的问题
    2. jieba对问题分词得到可能实体提及，收集问题中出现的字符
    3. get_entity_candidate() 直接从mention2ent.txt中得到候选实体
    4. evaluate_entity_precision() 计算候选实体的质量（error.json保存了没有得到实体的数据）
    5. get_entity_mention() 获得实体的提及
    6. get_entity_relation_pair() 得到实体关系对候选
    7. split_train_test() 将数据集分割成9:1（train:dev）
    '''
    start_time = datetime.now()
    # load our dict.txt
    jieba.load_userdict(OUR_JIEBA_DICT)
    print('loaded user dict: ', datetime.now() - start_time)
    seg = pkuseg.pkuseg()  # 默认初始化
    stopwords = json.load(open(STOPWORDS_PATH, 'r', encoding='utf-8'))

    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        i = 0  # 四行为一个问题的信息
        for line in file:
            i += 1
            if i == 1:
                question = re.findall(r'q[0-9]+:(.+)', line.rstrip('？\n').rstrip('?\n'))[0]
            elif i == 2:
                parse, sparql = sparql_parser(line)
            elif i == 3:
                answer = line.strip().split('\t')
                # 一个问题的信息收集完毕
                # 分词
                splited_question = [each for each in jieba.lcut(question, cut_all=True)
                                    if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
                splited_question += [each for each in seg.cut(question)
                                     if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
                # "XXXX" 大概率是entity
                for each in re.findall(r'"(.+)"', question):
                    splited_question.append(each)
                splited_question = sorted(list(set(splited_question)), key=lambda x: len(x), reverse=True) # 去重
                splited_question = [each for each in splited_question if each not in stopwords]

                data.append({
                    'question': question,
                    'tokens': splited_question,
                    'entity_candidates': {},
                    'parse': parse,
                    'answer': answer,
                    'sparql': sparql
                })
            elif i == 4:
                # 空行
                i = 0

    print('\ngetting entity candidates ...')
    data = get_entity_candidates(data)
    print('get_entity_candidates: ', datetime.now() - start_time)

    print('\nevaluating the ratio of entity existed in candidates ...')
    evaluate_entity_precision(data)
    print('evaluate_entity_precision: ', datetime.now() - start_time)

    print('\nget entity mention ...')
    get_entity_mention(data)
    print('get_entity_mention: ', datetime.now() - start_time)

    print('\ngetting entity-relation pair ...')
    data = get_entity_relation_pair(data)
    print('get_entity_relation_pair: ', datetime.now() - start_time)

    print('\nevaluating pair precision ... ')
    evaluate_pair_precision(data)
    print('evaluate_pair_precision: ', datetime.now() - start_time)

    # 只保存有用的信息
    new_data = []
    for node in data:
        new_data.append({
            'question': node['question'],
            'entity_candidates': node['entity_candidates'],
            'entity_mention': node['entity_mention'],
            'pair_candidates': node['pair_candidates'],
            'parse': node['parse'],
            'answer': node['answer'],
            'sparql': node['sparql']
        })
    data = new_data

    print('\nspliting data into train and test ...')
    split_train_test(data)
    print('split_train_test: ', datetime.now() - start_time)


def get_entity_candidates(data):
    starttime = datetime.now()
    mention2ent = json.load(open(PKUBASE_MENTION2ENT_COLLECTED, 'r', encoding='utf-8'))
    print('loaded mention2ent')
    num = 0
    count = 0
    for node in data:
        node['entity_candidates'] = {}

        # 提取日期，转化为entity
        date = re.findall(r'([0-9]+年[0-9]+月[0-9]+日|[0-9]+年[0-9]+月|[0-9]+月[0-9]+日)', node['question'])
        if date:
            temp = []
            for each in re.split(r'年|月|日', date[0]):
                if len(each) == 1:  # 补零
                    temp.append('0' + each)
                elif each != '':
                    temp.append(each)
            node['entity_candidates'][date[0]] = ['"' + '-'.join(temp) + '"']

        for token in node['tokens']:
            if token in mention2ent and token not in node['entity_candidates']:
                node['entity_candidates'][token] = mention2ent[token]
            else:
                # !todo 当mention2ent不包含时
                node['entity_candidates'][token] = ['<'+token+'>', '"'+token+'"']

        num += len(sum(list(node['entity_candidates'].values()), []))
        count += 1
        # if count % 100 == 0:
        #     print(f'{count} finished, {datetime.now()-starttime}')
    print('average entity candidates: ', num / len(data))
    print('data: %d, got entity candidates' % len(data))
    return data


def get_entity_mention(data):
    num = 0
    for node in data:
        node['entity_mention'] = {}
        entity = [each[0] for each in node['parse']]
        candidates = sorted(node['entity_candidates'].items(), key=lambda item: len(item[0]),
                            reverse=True)  # 按照mention的长度从长到短排序
        for e in entity:
            for item in candidates:
                if e in item[1]:  # mention, entity
                    node['entity_mention'][item[0]] = item[1]
                    break
        num += len(sum(list(node['entity_mention'].values()), []))

    print('average mention2entity candidates: ', num / len(data))
    print('data: %d, got mention2entity candidates' % len(data))


def get_entity_relation_pair(data):
    # 获取entity relation对候选

    def get_relation(query):
        results = sparql_query(query)
        relation = []
        for result in results['results']['bindings']:
            if 'relation' in result:
                relation.append(result['relation']['value'])
        return relation

    entity_relation = 'select distinct ?relation from ' + QUERY_GRAPH + ' where {{{entity} ?relation ?object}}'
    relation_entity = 'select distinct ?relation from ' + QUERY_GRAPH + ' where {{?subject ?relation {entity}}}'

    count = 0
    number = 0
    start_t = datetime.now()
    for node in data:
        number += 1
        # print(number, node['question'])
        if number % 100 == 0:
            print(number, ' got entity_relation pair ', datetime.now() - start_t)

        pair_candidates = {}
        entity_mention = list(node['entity_mention'].items())
        if 'entity_candidates' in node:
            entity_candidates = list(node['entity_candidates'].items())
            entity_mention += entity_candidates if len(entity_candidates) <= 2 else random.sample(entity_candidates, 2)

        for mention, es in entity_mention:
            pair_candidates[mention] = []
            for e in es:
                if any([c in e[1:-1] for c in [' ', '|', '<', '>', '"', '{', '}', '\\']]):  # sparql query报错了
                    continue
                # print(e)
                relations = get_relation(entity_relation.format(entity=e))
                for r in relations:
                    if any([c in r for c in [' ', '|', '<', '>', '"', '{', '}', '\\']]):
                        continue
                    pair_candidates[mention].append([e, '+', '<' + r + '>'])

                relations = get_relation(relation_entity.format(entity=e))
                for r in relations:
                    if any([c in r for c in [' ', '|', '<', '>', '"', '{', '}', '\\']]):
                        continue
                    pair_candidates[mention].append([e, '-', '<' + r + '>'])
        node['pair_candidates'] = pair_candidates

        for candidate in list(node['pair_candidates'].values()):
            count += len(candidate)
    print('average entity-relation pair number for each question is', count / len(data))
    return data


def split_train_test(data):
    # 将数据集分割成9：1的训练集和验证机
    def split(data, shuffle=False, ratio=0.1):
        length = len(data)
        if shuffle:
            offset = int(length * ratio)
            if length == 0 or offset < 1:
                return [], data
            random.shuffle(data)
            return data[offset:], data[:offset]  # 训练集、验证集
        else:
            train, dev = [], []
            distance = int(1 / ratio)
            for i in range(length):
                if i % distance:
                    train.append(data[i])
                else:
                    dev.append(data[i])
            return train, dev

    # train, dev = split(data, shuffle=True, ratio=0.1)
    train, dev = split(data, shuffle=False, ratio=0.1)

    print('the size of train and dev: %d, %d' % (len(train), len(dev)))
    json.dump(train, open(FILTER_TRAIN_DATA_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(dev, open(FILTER_DEV_DATA_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 如果发布了验证集的SPARQL等信息，需要稍作修改
    print('train_filer_prepare.py running ...')
    train_filter_prepare(TRAIN_DATA_PATH)
