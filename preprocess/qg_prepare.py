import json
import random
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(''))

from utils.configure import FILTER_TRAIN_DATA_PATH, FILTER_DEV_DATA_PATH, RANK_DEV_DATA_PATH, RANK_TRAIN_DATA_PATH, \
    QUERY_GRAPH
from utils.evaluate import evaluate_qg_precision
from utils.sparql import sparql_query


def get_entity_relation_pair(data_path):
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    for node in data:
        node['pair'] = {}

        # true_pair = []  # 文字
        # for p in node['parse']:
        #     pair = p[:3]
        #     true_pair.append(pair)
        candidates = [[key, v] for key, value in node['pair_candidates'].items() for v in value if len(v) == 3]

        # true_pair_ = []  # [[mention, entity]]
        # for cand in candidates:
        #     if cand[1] in true_pair:
        #         true_pair_.append(cand)

        candidates = candidates if len(candidates) < 10 else random.sample(candidates, k=10)
        # for pair in true_pair_:
        #     if pair not in candidates:
        #         candidates.append(pair)

        for cand in candidates:
            if cand[0] in node['pair']:
                node['pair'][cand[0]].append(cand[1])
            else:
                node['pair'][cand[0]] = [cand[1]]
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
