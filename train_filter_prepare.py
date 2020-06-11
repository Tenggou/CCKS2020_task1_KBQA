import json
import random
import re

import jieba
# from stanfordcorenlp import StanfordCoreNLP
from datetime import datetime

from SPARQLWrapper import JSON, SPARQLWrapper


def sparql_parser(line):
    '''
    20200516 解析sparql，得到entity literal value及其到答案节点的路径
    '''
    # 获得答案节点
    goal = re.findall(r'select[ |a-z]*(\?.+?) where .+', line.lower())[0]

    # path = list(filter(lambda x: not re.fullmatch(r' *', x), re.findall(r'.*where *{(.+)}.*', line.lower())[0].split('. ')))
    path = list(filter(lambda x: not re.fullmatch(r' *', x), re.findall(r'.*{(.+)}', line)[0].split('. ')))
    path_ = [] # 剩余的、分离的 ttls
    parser = []
    # 解析包含entity literal value的 ttls
    for ttl in path:
        splited_ttl = list(filter(lambda x: x, re.findall(r'\?.|<.+?>|".+?"', ttl)))
        if len(splited_ttl) == 2:
            if re.fullmatch('".+"', splited_ttl[1]):
                parser.append({
                    'entity': splited_ttl[1],
                    'entity_type': 'value',
                    'path': [splited_ttl],
                    'direction': 'r'
                })
        elif len(splited_ttl) == 3:
            if re.fullmatch('".+"', splited_ttl[0]):
                parser.append({
                    'entity': splited_ttl[0],
                    'entity_type': 'literal',
                    'path': [splited_ttl],
                    'direction': 'l'
                })
            elif re.fullmatch('".+"', splited_ttl[2]):
                parser.append({
                    'entity': splited_ttl[2],
                    'entity_type': 'literal',
                    'path': [splited_ttl],
                    'direction': 'r'
                })
            elif re.fullmatch('<.+>', splited_ttl[0]):
                parser.append({
                    'entity': splited_ttl[0],
                    'entity_type': 'entity',
                    'path': [splited_ttl],
                    'direction': 'l'
                })
            elif re.fullmatch('<.+>', splited_ttl[2]):
                parser.append({
                    'entity': splited_ttl[2],
                    'entity_type': 'entity',
                    'path': [splited_ttl],
                    'direction': 'r'
                })
            else:
        # 不包含entity literal value的ttl
                path_.append(splited_ttl)
        else:
            if splited_ttl and len(splited_ttl) <= 3:  # []、filter
                path_.append(splited_ttl)

    # 搜索从entity到答案节点的路径
    finished = 0
    while path_ and finished != len(parser):
        delete_ttls = []
        for node in parser:
            if node['direction'][-1] is 'l':
                next = node['path'][-1][2]
            elif node['direction'][-1] is 'r':
                next = node['path'][-1][0]
            if next == goal:
                finished += 1
                continue
            for ttl in path_:
                if next == ttl[0]:
                    node['path'].append(ttl)
                    node['direction'] += 'l'
                    delete_ttls.append(ttl)
                elif next == ttl[2]:
                    node['path'].append(ttl)
                    node['direction'] += 'r'
                    delete_ttls.append(ttl)
        for ttl in delete_ttls:
            if ttl in path_:
                path_.remove(ttl)
    # print('. '.join(path))
    # print(goal)
    # print(parser)
    # print(path_)
    return parser, line.strip()  # path, sparql


def txt2json_train():
    '''
    20200516
    1. sparql_parser 对原数据进行解析，获得问题、实体、实体到答案的路径、分词的问题
    2. jieba对问题分词得到可能实体提及，收集问题中出现的字符
    3. get_entity_candidate() 直接从mention2ent.txt中得到候选实体
    4. evaluate_entity_precision() 计算候选实体的质量（error.json保存了没有得到实体的数据）
    5. split_train_test() 将数据集分割成9:1（train:dev）
    '''
    path = 'data/ccks_2020_7_4_Data/task1-4_train_2020.txt'

    start_time = datetime.now()

    # 保存训练集的字
    characters = json.load(open('resources/characters.txt', 'r', encoding='utf-8'))
    # load our dict.txt
    jieba.load_userdict('resources/dict.txt')
    print('loaded user dict: ', datetime.now() - start_time)

    data = []
    with open(path, 'r', encoding='utf-8') as file:
        i = 0
        count = 0
        question = ''
        for line in file:
            i += 1
            if i == 1:
                question = re.findall(r'q[0-9]+:(.+)', line.rstrip('？\n').rstrip('?\n'))[0]
                characters += list(question)
                # print(re.findall(r'q([0-9]+):', line.strip())[0])
            elif i == 2:
                parse, sparql = sparql_parser(line)
            elif i == 3:
                answer = line.strip().split('\t')

                # count += 1

                # jieba分词
                splited_question = [word for word in jieba.lcut(question, cut_all=True) if word not in ['', ' ', ',', '?', '？']]
                # splited_question = [word for word in jieba.lcut(question) if word not in ['', ' ', ',', '?', '？']]

                data.append({
                    'question': question,
                    'tokens': splited_question,
                    # 'entity_mention': entity_mention,
                    'entity_candidates': {},
                    'parse': parse,
                    'answer': answer,
                    'sparql': sparql
                })
            elif i == 4:
                # 空行
                i = 0

    characters = list(set(characters))
    characters.append('UNK')  # 为未知的字符准备
    json.dump(characters, open('resources/characters.txt', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('data: %d, all finished, ' % len(data), datetime.now() - start_time)

    print('\ngetting entity candidates ...')
    data = get_entity_candidates(data)

    print('\nevaluating the ratio of entity existed in candidates ...')
    evaluate_entity_precision(data)

    print('\ngetting entity-relation pair ...')
    data = get_entity_relation_pair(data)

    print('\nspliting data into train and test ...')
    split_train_test(data)


def get_entity_candidates(data):
    mention2ent = json.load(open('resources/mention2ent.json', 'r', encoding='utf-8'))
    count = 0
    for node in data:
        node['entity_candidates'] = {}
        for token in node['tokens']:
            if token in mention2ent:
                node['entity_candidates'][token] = mention2ent[token]
            else:
                # ！todo 当mention没有直接存在mention2ent.txt中时（提升点）
                if token.lower() in mention2ent:
                    node['entity_candidates'][token.lower()] = mention2ent[token.lower()]
                if token.upper() in mention2ent:
                    node['entity_candidates'][token.upper()] = mention2ent[token.upper()]
        for candidate in list(node['entity_candidates'].values()):
            count += len(candidate)

        # 让entity按parse的长度下降排序
        node['parse'] = sorted(node['parse'], key=lambda item: len(item['direction']), reverse=True)

    print('average entity candidates: ', count/len(data))  # 45
    print('data: %d, got entity candidates' % len(data))
    return data


def evaluate_entity_precision(data):
    '''
    计算（实体提及）实体候选的准确度
    '''
    rate = 0.0
    error = []
    for node in data:
        r = 0
        entity = [each['entity'] for each in node['parse']]
        cadidate = sum(list(node['entity_candidates'].values()), [])  # 多维list转为一维
        # print(cadidate)
        for e in entity:
            if e in cadidate:
                r += 1
        if len(entity) != 0:
            rate += (r/len(entity))
        if len(entity) == 0 or r != len(entity):
            error.append(node)
    print('the ratio of true entities exist in candidates: ', rate/len(data))  # 0.9339166666666666
    json.dump(error, open('data/error.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def get_entity_relation_pair(data):
    from sparql import endpoint

    def get_relation(query):
        sparql = SPARQLWrapper(endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        # sparql.setTimeout(5)
        results = sparql.query().convert()
        relation = []
        for result in results['results']['bindings']:
            if 'relation' in result:
                relation.append(result['relation']['value'])
        return relation

    entity_relation = 'select distinct ?relation from <http://pkubase.cn> where {{{entity} ?relation ?object}}'
    relation_entity = 'select distinct ?relation from <http://pkubase.cn> where {{?subject ?relation {entity}}}'

    count = 0
    number = 0
    for node in data:
        number += 1
        # print(number, node['question'])
        if number % 100 == 0:
            print(number, ' got entity_relation pair')

        pair_candidates = {}
        for mention, es in node['entity_candidates'].items():  # entity_candidates
            pair_candidates[mention] = []
            for e in es:
                if any([c in e[1:-1] for c in [' ', '|', '<', '>', '"', '{', '}', '\\']]):  # sparql query报错了
                    continue
                # print(e)
                relations = get_relation(entity_relation.format(entity=e))
                for r in relations:
                    pair_candidates[mention].append([[e, '<'+r+'>'], 'er'])

                relations = get_relation(relation_entity.format(entity=e))
                for r in relations:
                    pair_candidates[mention].append([['<'+r+'>', e], 're'])
        node['pair_candidates'] = pair_candidates

        for candidate in list(node['pair_candidates'].values()):
            count += len(candidate)
    print('average entity-relation pair number for each question is', count/len(data))  # 928.45

    print('evaluating precision ... ')
    evaluate_pair_precision(data)
    return data


def evaluate_pair_precision(data):
    rate = 0.0
    # error = []
    for node in data:
        r = 0
        true_pair = []  # 文字
        for p in node['parse']:
            if p['direction'][0] == 'l':
                true_pair.append([p['path'][0][:2], 'er'])
            else:
                true_pair.append([p['path'][0][1:], 're'])
        candidate = sum(list(node['pair_candidates'].values()), [])  # 多维list转为一维
        # print(candidate)
        for e in true_pair:
            if e in candidate:
                r += 1
        if len(true_pair) != 0:
            rate += (r / len(true_pair))

        # print(json.dumps(node, indent=4, ensure_ascii=False))
        # print(candidate)
        # print(json.dumps(node['parse'], indent=4, ensure_ascii=False))
        # print(true_pair)

    print('the ratio of true entity_relation pair exist in candidates: ', rate / len(data))  # 0.89


def split_train_test(data):
    # 将数据集分割成9：1的训练集和验证机
    def split(data, shuffle=False, ratio=0.1):
        length = len(data)
        offset = int(length * ratio)
        if length == 0 or offset < 1:
            return [], data
        if shuffle:
            random.shuffle(data)
        return data[offset:], data[:offset]  # 训练集、验证集
    train, dev = split(data, shuffle=True, ratio=0.1)

    print('the size of train and dev: %d, %d' % (len(train), len(dev)))
    json.dump(train, open('data/train_filter/train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(dev, open('data/train_filter/dev.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    txt2json_train()

    # data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8')) + \
    #        json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    # get_entity_relation_pair(data)
