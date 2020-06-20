import json
import re

import jieba
import torch
from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader

from models import Model
from train_filter import load_vocabulary, FilterDataset, get_topk_candidate
from train_filter_prepare import get_entity_candidates, get_entity_relation_pair
from train_rank import test_collate, RankDataset
from train_rank_prepare import get_query_graph

endpoint = 'http://10.201.180.179:8890/sparql'


def build_sparql(parse):
    # 根据之前解析的结果重新组建成SPARQL
    sparql = 'select distinct ?x from <http://pkubase.cn> where {{{query}}}'
    query = []
    answer_ttl = {}
    answer = '?x'

    for i in range(len(parse)):
        p = parse[i]

        if re.fullmatch('<.+?>|".+?"', p[0]) and len(p) % 2 == 1:  # 实体出发的
            last_v = p[0]
            p = np.array(p[1:]).reshape((-1, 2)).tolist()  # 一维转化为二维
        elif not re.fullmatch('<.+?>|".+?"', p[0]) and len(p) % 2 == 0:  # 不是实体开始的
            last_v = '?v' + str(i)
            p = np.array(p).reshape((-1, 2)).tolist()  # 一维转化为二维
        else:
            continue

        for j in range(len(p)):  # 构建path
            v = '?v'+str(i)+str(j)
            if p[j][0] == '+':  #
                query.append(' '.join([last_v, p[j][1], v]))
            else:
                query.append(' '.join([v, p[j][1], last_v]))
            last_v = v
        query[-1] = query[-1].replace(last_v, answer)  # 替换答案变量

        # 标记答案变量的之前一个变量
        key = None
        value = list(filter(lambda x: x, re.findall(r'\?.+? |<.+?>|".+?"', query[-1])))  # 这里变量不止一个字符！！
        if p[-1][0] == '+' and '?' in value[0]:  # 确定关系方向，以及保证实体不被替换
            key = ' '.join([p[-1][1], answer])
            value = value[0].strip()
        elif p[-1][0] == '-' and '?' in value[-1]:
            key = ' '.join([answer, p[-1][1]])
            value = value[-1].strip()
        if key and key in answer_ttl:
            answer_ttl[key].append(value)
        elif key and key not in answer_ttl:
            answer_ttl[key] = [value]
    query.append(' ')
    query = ' . '.join(query)

    # print(answer_ttl)
    # print(query)
    # 替换答案变量的之前一个变量
    count = 0
    for values in answer_ttl.values():
        v = '?v_' + str(count)
        for value in values:
            query = query.replace(value, v)
        count += 1
        # print(query)

    return sparql.format(query=query)


def get_answer(query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30)
    results = sparql.query().convert()
    answer = []
    for result in results['results']['bindings']:
        if 'x' in result:
            if result['x']['type'] == 'uri':
                answer.append('<'+result['x']['value']+'>')
            elif result['x']['type'] == 'literal':
                answer.append('"'+result['x']['value']+'"')
    return answer


def f1_score(true, pred):
    count = 0
    for t in true:
        if t in pred:
            count += 1
    # print(true)
    # print(pred)
    precision = count/len(pred) if len(pred) else 0  # 预测的准确率
    recall = count/len(true) if len(pred) else 0  # 召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    return precision, recall, f1


def parse_result():
    '''
    测试重建的sparql方法的效果
    '''
    start_time = datetime.now()
    data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8')) + \
           json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    count = 0
    precision = 0
    recall = 0
    f1 = 0
    error = []
    for node in data:
        # 塑造sparql
        count += 1
        query = build_sparql(node['parse'])

        # print(node['parse'])
        # print(query)

        answer = get_answer(query)
        result = f1_score(node['answer'], answer)

        if any([r != 1 for r in result]):
            node['pred_answer'] = answer
            error.append(node)

        precision += result[0]
        recall += result[1]
        f1 += result[2]

        if count % 100 == 0:
            print(count, ' finished')
            # break
    precision /= len(data)
    recall /= len(data)
    f1 /= len(data)
    # 自己的pkubase-clean，准确率： 0.934961, 召回率：0.947956, F1-score：0.937219
    print('准确率： %f, 召回率：%f, F1-score：%f' % (precision, recall, f1))
    json.dump(error, open('data/parser_answer_error.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(datetime.now() - start_time)


def dev_result():
    start_time = datetime.now()

    model_name = 'lstm'
    device = 'cuda:1'
    data = json.load(open('data/train_rank/dev.json', 'r', encoding='utf-8'))

    vocabulary = load_vocabulary()
    model = Model(device=device, size_vocabulary=len(vocabulary)+1, model_name=model_name)
    print('load model ...')
    model.load('model/rank/' + model_name)
    print('loading data ...')
    data = DataLoader(
        RankDataset(data=data, device=device, is_train=False, model_name=model_name, vocabulary=vocabulary),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    new_data = []
    for _, item in enumerate(data):
        index = model.predict_rank(item)
        pred = item['qg_'][index]
        new_data.append({
            'question': item['origin_data']['question'],
            'pred': pred,  # query graph
            'answer': item['origin_data']['answer'],
            'sparql': item['origin_data']['sparql'],
            'parse': item['origin_data']['parse']
        })
    print('predicted all query graph ', datetime.now() - start_time)

    count = 0
    precision = 0
    recall = 0
    f1 = 0
    # error = []
    for node in new_data:
        # 塑造sparql
        count += 1
        query = build_sparql(node['pred'])

        # print(node['pred'])
        # print(query)

        answer = get_answer(query)
        result = f1_score(node['answer'], answer)

        # if any([r != 1 for r in result]):
        #     node['pred_answer'] = answer
        #     error.append(node)

        precision += result[0]
        recall += result[1]
        f1 += result[2]

        if count % 100 == 0:
            print(count, ' finished')
            # break
    precision /= len(data)
    recall /= len(data)
    f1 /= len(data)
    # 自己的pkubase-clean，准确率： 0.618728, 召回率：0.640037, F1-score：0.619357
    print('准确率： %f, 召回率：%f, F1-score：%f' % (precision, recall, f1))
    # json.dump(error, open('data/parser_answer_error.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(datetime.now() - start_time)


def test_result(path):
    """
    完整的测试流程：从问题到查询图
    :param path:
    :return:
    """
    start_time = datetime.now()
    k = 10
    model_name = 'lstm'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # 检测GPU

    # jieba.load_userdict('resources/dict.txt')
    # print('loaded user dict: ', datetime.now() - start_time)
    #
    # data = []
    # with open(path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         question = re.findall(r'q[0-9]+:(.+)', line.rstrip('？\n').rstrip('?\n'))[0]
    #         splited_question = [word for word in jieba.lcut(question, cut_all=True) if
    #                             word not in ['', ' ', ',', '?', '？']]
    #         data.append({
    #             'question': question,
    #             'tokens': splited_question,
    #             'entity_candidates': {},
    #         })
    #
    # print('\ngetting entity candidates ...')
    # data = get_entity_candidates(data)
    # json.dump(data, open(path+model_name+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    #
    # print('\ngetting entity-relation pair ...')
    # data = get_entity_relation_pair(data)
    # json.dump(data, open(path+model_name+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    #
    # '''
    # filter  ##################################
    # '''
    # if torch.cuda.is_available():
    #     from torch.backends import cudnn
    #     cudnn.benchmark = True
    # print('device name ', device)
    #
    # vocabulary = load_vocabulary()
    # model = Model(model_name=model_name, device=device, size_vocabulary=len(vocabulary) + 1)  # +1 为pad值，index=0
    #
    # print('load model ...')
    # model.load('model/filter/' + model_name + str(k))
    # print('loading data ...')
    # data = DataLoader(FilterDataset(data, vocabulary, device, is_train=False, model_name=model_name, is_test=True),
    #                   batch_size=1, num_workers=0, collate_fn=test_collate)
    #
    # new_data = []
    # print('filting entity relation pair ...')
    # for i, item in enumerate(data):
    #     index = model.predict_filter(item, k)
    #     pair = get_topk_candidate(index, item)  # dict: key value
    #     new_data.append({
    #         'question': item['origin_data']['question'],
    #         'pair': pair,  # e +/- r
    #     })
    # print('filter pair has finished !', datetime.now() - start_time)
    # json.dump(new_data, open(path+model_name+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    #
    # '''
    # get query graph candidates #########
    # '''
    # print('getting query graph ... ')
    # new_data = get_query_graph(new_data)
    # json.dump(new_data, open(path+model_name+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    rank ################################
    '''
    # vocabulary = load_vocabulary()
    # model = Model(device=device, size_vocabulary=len(vocabulary) + 1, model_name=model_name)
    # print('load model ...')
    # model.load('model/rank/' + model_name)
    # print('loading data ...')
    # data = DataLoader(
    #     RankDataset(data=new_data, device=device, is_train=False, model_name=model_name, vocabulary=vocabulary, is_test=True),
    #     batch_size=1, num_workers=0, collate_fn=test_collate)
    #
    # print('ranking ...')
    # new_data = []
    # for _, item in enumerate(data):
    #     index = model.predict_rank(item)
    #     pred = item['qg_'][index]
    #     new_data.append({
    #         'question': item['origin_data']['question'],
    #         'pred': pred,  # query graph
    #     })
    # print('predicted all query graph ', datetime.now() - start_time)
    # json.dump(new_data, open(path+model_name+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    get answer  #############################
    '''
    new_data = json.load(open(path+model_name+'.json', 'r', encoding='utf-8'))

    print('getting answer ...')
    with open(path+model_name+'.answer.txt', 'w', encoding='utf-8') as answer_file:
        count = 0
        for node in new_data:
            # 塑造sparql
            count += 1
            query = build_sparql(node['pred'])
            answer = get_answer(query)
            answer_file.write('\t'.join(answer)+'\n')

            if count % 100 == 0:
                print(count, ' finished')
                # break
    print('got answers ', datetime.now() - start_time)


if __name__ == '__main__':
    path = 'data/ccks_2020_7_4_Data/task1-4_valid_2020.questions'
    # parse_result()

    # dev_result()

    test_result(path)
