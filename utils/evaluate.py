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
def get_topk_candidate(index, data):
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


def evaluate_filter(model, data):
    print('evaluating filter ...')
    start_time = datetime.now()
    accuracy = 0
    length = 0

    def calulate_acc(true, pred):
        if len(true) == 0:
            return 0
        acc = 0
        pred = sum(list(pred.values()), [])
        for v in true:
            if v in pred:
                acc += 1
        return acc / len(true)

    for i, item in enumerate(data):
        index = model.predict(item)
        acc = calulate_acc(item['true_pair'], get_topk_candidate(index, item))
        accuracy += acc
        length += 1
    accuracy /= length
    print('accuracy is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy


def evaluate_rank(model, data):
    print('evaluating rank ...')
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
    print('accuracy is %.4f ' % accuracy)
    print('evaluate time is ', datetime.now() - start_time)
    return accuracy
