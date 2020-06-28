import json
import re
from datetime import datetime

import torch
from SPARQLWrapper import SPARQLWrapper, JSON
from torch.utils.data import DataLoader

from models import Model

from train_filter import load_vocabulary, FilterDataset, test_collate, get_topk_candidate
from train_filter_prepare import endpoint


def get_entity_relation_pair(data_path):
    '''
    利用训练的filter获取topk实体关系对，并获取必要信息
    :param data_path:
    :return:
    '''
    new_data = []
    k = 10
    model_name = 'bert'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检测GPU

    if torch.cuda.is_available():
        from torch.backends import cudnn
        cudnn.benchmark = True
    print('device name ', device)

    vocabulary = load_vocabulary()
    model = Model(model_name=model_name, device=device, size_vocabulary=len(vocabulary) + 1)  # +1 为pad值，index=0

    print('load model ...')
    model.load('model/filter/'+model_name+str(k))
    print('loading data ...')
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    data = DataLoader(FilterDataset(data, vocabulary, device, is_train=False, model_name=model_name),
                           batch_size=1, num_workers=0, collate_fn=test_collate)

    def calulate_acc(true, pred):
        if len(true) == 0:
            return 0
        acc = 0
        pred = sum(list(pred.values()), [])
        for v in true:
            if v in pred:
                acc += 1
        return acc / len(true)
    accuracy = 0
    length = 0
    for i, item in enumerate(data):
        index = model.predict_filter(item, k)
        pair = get_topk_candidate(index, item)  # dict: key value
        accuracy += calulate_acc(item['true_pair'], pair)
        length += 1
        new_data.append({
            'question': item['origin_data']['question'],
            'pair': pair,  # e +/- r
            'answer': item['origin_data']['answer'],
            'sparql': item['origin_data']['sparql'],
            'parse': item['origin_data']['parse']
        })
    accuracy /= length
    print('accuracy is %.3f ' % accuracy)
    return new_data


def get_query_graph(data):
    # 获取查询图候选

    def get_list(pair):
        candidate = sum(list(pair.values()), [])  # 二维变一维
        return candidate

    def two_hop_one_entity(candidate):
        ll = 'select distinct ?r from <http://pkubase.cn> where {{{} {} ?v . ?v ?r ?x .}} limit 100'
        lr = 'select distinct ?r from <http://pkubase.cn> where {{{} {} ?v . ?x ?r ?v .}} limit 100'
        rr = 'select distinct ?r from <http://pkubase.cn> where {{?v {} {} . ?x ?r ?v .}} limit 100'
        rl = 'select distinct ?r from <http://pkubase.cn> where {{?v {} {} . ?v ?r ?x .}} limit 100'

        def query(q):
            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(q)
            sparql.setReturnFormat(JSON)
            # sparql.setTimeout(300)
            results = sparql.query().convert()
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
                    qg.append([c + ['+', '<'+each+'>']])

                q = lr.format(c[0], c[-1])
                r = query(q)
                for each in r:
                    qg.append([c + ['-', '<'+each+'>']])
            elif c[index] == '-':
                q = rr.format(c[-1], c[0])
                r = query(q)
                for each in r:
                    qg.append([c + ['-', '<'+each+'>']])

                q = rl.format(c[-1], c[0])
                r = query(q)
                for each in r:
                    qg.append([c + ['+', '<'+each+'>']])
        return qg

    def one_hop_two_entity(pair):
        query_temp = 'ask from <http://pkubase.cn> where {{{}}}'
        ll = '{} {} ?x . {} {} ?x . '
        lr = '{} {} ?x . ?x {} {} . '
        rl = '?x {} {} . {} {} ?x . '
        rr = '?x {} {} . ?x {} {} . '

        def get_boolean(query):
            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            # sparql.setTimeout(5)
            results = sparql.query().convert()
            return results['boolean']

        qg = []
        pair = list(pair.values())
        if len(pair) < 2:
            return qg

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
                            qg.append([pre, pos, j])
        return qg

    def one_hop_three_entity(pair, temp_1h2e):

        def get_boolean(query):
            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            # sparql.setTimeout(5)
            results = sparql.query().convert()
            return results['boolean']

        query_temp = 'ask from <http://pkubase.cn> where {{{}}}'
        l = '{} {} ?x'
        r = '?x {} {}'

        qg = []
        pair = list(pair.values())
        for t in temp_1h2e:
            for p in pair[t[-1]+1:]:
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
        if count % 10 == 0:
            print(count, ' got query graph ', datetime.now()-start_time)
        # print(count, ' got query graph ', datetime.now()-start_time)
    return data


def evaluate_qg_precision(data):
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


def train_rank_prepare():
    '''
    获取训练rank数据
    :return:
    '''
    start_time = datetime.now()
    train_data_path = 'data/train_filter/train.json'
    dev_data_path = 'data/train_filter/dev.json'

    dev_data = get_entity_relation_pair(dev_data_path)  # acc: 0.833
    torch.cuda.empty_cache()
    print('dev_data got entity_relation pair ', datetime.now()-start_time)

    dev_data = get_query_graph(dev_data)
    print('got query graph pair ', datetime.now()-start_time)

    # dev_data = json.load(open('data/train_rank/dev.json', 'r', encoding='utf-8'))
    evaluate_qg_precision(dev_data)  # ratio: 0.748 num: 324

    json.dump(dev_data, open('data/train_rank/dev.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    #################################

    train_data = get_entity_relation_pair(train_data_path)  # acc: 0.943
    torch.cuda.empty_cache()
    print('train_data got entity_relation pair ', datetime.now()-start_time)
    train_data = get_query_graph(train_data)
    print('train_data got query graph pair ', datetime.now() - start_time)  # 58.00

    # train_data = json.load(open('data/train_rank/train.json', 'r', encoding='utf-8'))
    evaluate_qg_precision(train_data)  # ratio: 0.838 num: 328

    json.dump(train_data, open('data/train_rank/train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    train_rank_prepare()
    # dev_data = json.load(open('data/train_rank/dev.json', 'r', encoding='utf-8'))
    # # 测试集上限 0.748
    # evaluate_qg_precision(dev_data)