import json
from datetime import datetime

from utils.configure import ENTITY_ERROR_PATH


# 准备数据
def evaluate_entity_precision(data):
    '''
    计算（实体提及）实体候选精确度
    '''
    rate = 0.0
    error = []
    for node in data:
        r = 0
        entity = [each[0] for each in node['parse']]  # SPARQL解析的结果中获取每个问题的entity
        cadidate = sum(list(node['entity_candidates'].values()), [])  # 多维list转为一维
        # print(cadidate)
        for e in entity:
            if e in cadidate:
                r += 1
        if len(entity) != 0:
            rate += (r/len(entity))
        if len(entity) == 0 or r != len(entity):
            error.append(node)
    print('the ratio of true entities exist in candidates: ', rate/len(data))
    json.dump(error, open(ENTITY_ERROR_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def evaluate_pair_precision(data):
    # 测试entity relation对候选精确度
    rate = 0.0
    for node in data:
        r = 0
        true_pair = []  # 文字
        for p in node['parse']:
            true_pair.append(p[:3])
        candidate = sum(list(node['pair_candidates'].values()), [])  # 多维list转为一维
        for e in true_pair:  # 不同 mention 有相同的实体
            if e in candidate:
                r += 1
        if len(true_pair) != 0:
            rate += (r / len(true_pair))
    print('the ratio of true entity_relation pair exist in candidates: ', rate / len(data))


def evaluate_qg_precision(data):
    # 测试查询图候选的精确度
    print('evaluating ...')
    # start_time = datetime.now()
    count = 0
    num = 0
    error = []
    for node in data:
        flag = 0
        candidates = sum(list(node['qg_candidates'].values()), [])
        for each in candidates:
            if len(each) == len(node['parse']) and all([e in each for e in node['parse']]):
                count += 1
                flag = 1
                break  # 不同的mention存在相同的实体
        if flag == 0:
            error.append(node['parse'])  # 出错检测
        num += len(candidates)
    print('the ratio of query graph in candidates is %.3f' % (count/len(data)))
    print('the average number of query graph is %.3f ' % (num/len(data)))
    return error


# 训练/测试
def f1_score(true, pred):
    count = 0
    for t in true:
        if t in pred:
            count += 1
    # print(true)
    # print(pred)
    precision = count/len(pred) if len(pred) else 0  # 预测的准确率
    recall = count/len(true) if len(true) else 0  # 召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    return precision, recall, f1


def get_topk_pair(index, data):
    # get topk candidates and sort into mentions。
    # 根据索引取得对应的候选并归类到相应提及中。
    pair = {}
    for i in index:
        if any([char in data['pair_'][i][0][1:-1] or char in data['pair_'][i][-1][1:-1] for char in
                [' ', '|', '<', '>', '"', '{', '}', '\\']]):  # sparql query报错了
            continue
        if data['mention'][i] not in pair:
            pair[data['mention'][i]] = [data['pair_'][i]]
        elif data['mention'][i] in pair:
            pair[data['mention'][i]].append(data['pair_'][i])
    return pair


def get_topk_entity(index, data):
    # get topk candidates and sort into mentions。
    # 根据索引取得对应的候选并归类到相应提及中。
    entity = {}
    for i in index:
        if any([char in data['entity'][i][1:-1] for char in [' ', '|', '<', '>', '"', '{', '}', '\\']]):  # sparql query报错了
            continue
        if data['mention'][i] not in entity:
            entity[data['mention'][i]] = [data['entity'][i]]
        elif data['mention'][i] in entity:
            entity[data['mention'][i]].append(data['entity'][i])
    return entity


def get_mentions(question, tags):
    question = [each for each in question]
    for i in range(len(question)):
        if tags[i] == 0:
            question[i] = ' '
    return [s for s in ''.join(question).split() if s]


def evaluate_pair(model, data):
    print('evaluating pair ...')
    start_time = datetime.now()
    accuracy = 0
    length = 0

    for i, item in enumerate(data):
        index = model.predict(item)
        acc = f1_score(item['true_pair'], [item['pair_'][i] for i in index])[1]
        accuracy += acc
        length += 1
    accuracy /= length
    print('pair recall is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


def evaluate_qg(model, data):
    print('evaluating qg ...')
    start_time = datetime.now()
    count = 0
    length = 0
    for _, item in enumerate(data):
        length += 1
        index = model.predict(item)
        pred = item['qg_'][index]
        if len(pred) == len(item['parse']) and all([e in pred for e in item['parse']]):
            count += 1
    accuracy = count / length
    print('count: %d, length: %d' % (count, length))
    print('qg accuracy is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


def evaluate_ner(model, data):
    print('evaluating ner ...')
    start_time = datetime.now()
    count = 0
    length = 0
    for _, item in enumerate(data):
        length += 1
        tags = model.predict(item)
        # print(tags, item['mention'], item['question'], item['label'])
        count += f1_score(item['mention'], get_mentions(item['question'], tags[0]))[1]

    accuracy = count / length
    # print('count: %d, length: %d' % (count, length))
    print('ner accuracy is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


def evaluate_entity(model, data):
    print('evaluating entity ...')
    start_time = datetime.now()
    accuracy = 0
    length = 0

    for i, item in enumerate(data):
        index = model.predict(item)
        acc = f1_score(item['true_entity'],[item['entity'][i] for i in index])[1]
        accuracy += acc
        length += 1
    accuracy /= length
    print('entity recall is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy