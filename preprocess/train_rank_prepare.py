import json
import random
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(''))

import torch
from torch.utils.data import DataLoader

from model.model import Model
from utils.dataset import FilterDataset, test_collate

from utils.configure import FILTER_TRAIN_DATA_PATH, FILTER_DEV_DATA_PATH, RANK_DEV_DATA_PATH, RANK_TRAIN_DATA_PATH, \
    K_PAIR, QUERY_GRAPH
from utils.evaluate import evaluate_qg_precision, get_topk_candidate
from utils.sparql import sparql_query


def get_entity_relation_pair(data_path):
    # # 效果不好
    # for node in data:
    #     node['pair'] = {}
    #     keys_candidates = list(node['pair_candidates'].keys())
    #     keys = random.sample(keys_candidates, int(K_PAIR / 2)) if len(keys_candidates) > int(
    #         K_PAIR / 2) else keys_candidates
    #     for key in keys:
    #         node['pair'][key] = random.sample(node['pair_candidates'][key], 2) if len(
    #             node['pair_candidates'][key]) > 2 else node['pair_candidates'][key]

    start_time = datetime.now()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检测GPU
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    '''
    filter  ##################################
    '''
    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
    print('device name is ', device)

    model = Model(device=device, component='filter', model_path='parameter/filter.' + str(K_PAIR) + '.model')
    print('load model ...')
    model.load()
    print("loading data ...")
    data = DataLoader(
        FilterDataset(data=data, device=device, is_test=True, is_train=False),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    new_data = []
    print('filting entity relation pair ...')
    for i, item in enumerate(data):
        index = model.predict_filter(item)
        pair = get_topk_candidate(index, item)  # dict: key value
        new_data.append({
            'question': item['origin_data']['question'],
            'pair': pair,  # e +/- r
            'answer': item['origin_data']['answer'],
            'sparql': item['origin_data']['sparql'],
            'parse': item['origin_data']['parse']
        })
    data = new_data
    print('filter pair has finished !', datetime.now() - start_time)
    return data


def get_query_graph(data):
    # 获取查询图候选

    def get_list(pair):
        return sum(list(pair.values()), [])  # 二维变一维

    def two_hop_one_entity(candidate):
        ll = 'select distinct ?r from ' + QUERY_GRAPH + ' where {{{} {} ?v . ?v ?r ?x .}} limit 100'
        lr = 'select distinct ?r from ' + QUERY_GRAPH + ' where {{{} {} ?v . ?x ?r ?v .}} limit 100'
        rr = 'select distinct ?r from ' + QUERY_GRAPH + ' where {{?v {} {} . ?x ?r ?v .}} limit 100'
        rl = 'select distinct ?r from ' + QUERY_GRAPH + ' where {{?v {} {} . ?v ?r ?x .}} limit 100'

        def query(q):
            results = sparql_query(q)
            relation = []
            for result in results['results']['bindings']:
                if 'r' in result:
                    relation.append(result['r']['value'])
            return relation

        qg = []
        for c in candidate:
            index = 1
            if c[index] == '+':
                q = ll.format(c[0], c[-1])
                r = query(q)
                for each in r:
                    qg.append([c + ['+', '<' + each + '>']])

                q = lr.format(c[0], c[-1])
                r = query(q)
                for each in r:
                    qg.append([c + ['-', '<' + each + '>']])
            elif c[index] == '-':
                q = rr.format(c[-1], c[0])
                r = query(q)
                for each in r:
                    qg.append([c + ['-', '<' + each + '>']])

                q = rl.format(c[-1], c[0])
                r = query(q)
                for each in r:
                    qg.append([c + ['+', '<' + each + '>']])
        return qg

    def one_hop_two_entity(pair):
        query_temp = 'ask from ' + QUERY_GRAPH + ' where {{{}}}'
        ll = '{} {} ?x . {} {} ?x . '
        lr = '{} {} ?x . ?x {} {} . '
        rl = '?x {} {} . {} {} ?x . '
        rr = '?x {} {} . ?x {} {} . '

        def get_boolean(query):
            return sparql_query(query)['boolean']

        qg = []
        pair = list(pair.values())
        if len(pair) < 2:
            return qg

        # 单跳查询路径间是否能够两两组合，保证同一个mention下的路径不会被组合
        for i in range(len(pair)):
            for pre in pair[i]:
                for j in range(i + 1, len(pair)):
                    for pos in pair[j]:
                        if pre[1] == '+' and pos[1] == '+':
                            query = ll.format(pre[0], pre[-1], pos[0], pos[-1])
                        elif pre[1] == '+' and pos[1] == '-':
                            query = lr.format(pre[0], pre[-1], pos[-1], pos[0])
                        elif pre[1] == '-' and pos[1] == '+':
                            query = rl.format(pre[-1], pre[0], pos[0], pos[-1])
                        elif pre[1] == '-' and pos[1] == '-':
                            query = rr.format(pre[-1], pre[0], pos[-1], pos[0])
                        else:
                            continue
                        if get_boolean(query_temp.format(query)):
                            qg.append([pre, pos, j])  # j是为了收集1h3e
        return qg

    def one_hop_three_entity(pair, temp_1h2e):

        def get_boolean(query):
            return sparql_query(query)['boolean']

        query_temp = 'ask from <http://pkubase.cn> where {{{}}}'
        l = '{} {} ?x'
        r = '?x {} {}'

        qg = []
        pair = list(pair.values())
        for t in temp_1h2e:
            for p in pair[t[-1] + 1:]:
                for each in p:
                    query = []
                    if t[0][1] == '+':
                        query.append(l.format(t[0][0], t[0][-1]))
                    elif t[0][1] == '-':
                        query.append(r.format(t[0][-1], t[0][0]))
                    if t[1][1] == '+':
                        query.append(l.format(t[1][0], t[1][-1]))
                    elif t[1][1] == '-':
                        query.append(r.format(t[1][-1], t[1][0]))
                    if each[1] == '+':
                        query.append(l.format(each[0], each[-1]))
                    elif each[1] == '-':
                        query.append(r.format(each[-1], each[0]))
                    query.append(' ')
                    query = ' . '.join(query)

                    if get_boolean(query_temp.format(query)):
                        qg.append([t[0], t[1], each])
        return qg

    start_time = datetime.now()
    count = 0
    # hop实体距离答案变量的跳数，entity实体数量
    for node in data:
        candidate = get_list(node['pair'])
        qg = {}
        qg['1h1e'] = [[c] for c in candidate]  # l / r
        # print(qg['1h1e'])
        qg['2h1e'] = two_hop_one_entity(candidate)  # lr / ll / rl / rr
        # print(qg['2h1e'])
        temp_1h2e = one_hop_two_entity(node['pair'])  # r_r / l_l / r_l / l_r
        qg['1h2e'] = [t[:2] for t in temp_1h2e]
        # print(qg['1h2e'])
        qg['1h3e'] = one_hop_three_entity(node['pair'], temp_1h2e)  # l_l_l / r_r_r
        # print(qg['1h3e'])
        node['qg_candidates'] = qg
        count += 1
        if count % 100 == 0:
            print(count, ' got query graph ', datetime.now() - start_time)
        # print(count, ' got query graph ', datetime.now()-start_time)
    return data


def train_rank_prepare():
    '''
    获取训练rank数据
    :return:
    '''
    start_time = datetime.now()

    dev_data = get_entity_relation_pair(FILTER_DEV_DATA_PATH)
    print('dev_data got entity_relation pair ', datetime.now() - start_time)

    dev_data = get_query_graph(dev_data)
    print('got query graph pair ', datetime.now() - start_time)

    evaluate_qg_precision(dev_data)

    json.dump(dev_data, open(RANK_DEV_DATA_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    #################################

    train_data = get_entity_relation_pair(FILTER_TRAIN_DATA_PATH)
    print('train_data got entity_relation pair ', datetime.now() - start_time)
    train_data = get_query_graph(train_data)
    print('train_data got query graph pair ', datetime.now() - start_time)  # 58.00

    evaluate_qg_precision(train_data)

    json.dump(train_data, open(RANK_TRAIN_DATA_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    print('train_rank_prepare.py running ... ')
    train_rank_prepare()
