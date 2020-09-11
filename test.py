import argparse
import ast
import json
import re
from datetime import datetime

import jieba
import pkuseg
import torch
from torch.utils.data import DataLoader

from model.model import Model
from preprocess.entity_and_pair_prepare import get_entity_candidates, get_entity_relation_pair
from preprocess.qg_prepare import get_query_graph
from utils.configure import TEST_DATA_PATH, OUR_JIEBA_DICT, STOPWORDS_PATH, FILTER_DEV_DATA_PATH
from utils.dataset import test_collate, EntityDataset, PairDataset, QGDataset, NERDataset
from utils.evaluate import get_topk_entity, get_topk_pair, get_mentions, f1_score
from utils.sparql import sparql_query, build_sparql


def get_answer(query):
    results = sparql_query(query)
    answer = []
    for result in results['results']['bindings']:
        if 'x' in result:
            if result['x']['type'] == 'uri':
                answer.append('<' + result['x']['value'] + '>')
            elif result['x']['type'] == 'literal':
                answer.append('"' + result['x']['value'] + '"')
    return answer


def result(path, data):
    """
    完整的测试流程：从问题到查询图
    :param path:
    :return:
    """

    start_time = datetime.now()
    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')  # 检测GPU

    '''
    entity  ##################################
    '''
    if k_entity == -1:
        print('\ngetting entity candidates ...')
        data = get_entity_candidates(data)
        # print(data[0]['entity_candidates'])
        for node in data:
            node['entity_mention'] = node.pop('entity_candidates')
            node['origin_data'] = {
                'question': node['question'],
                'answer': node['answer']
            }
        # print(data[0]['entity_mention'])
        json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print('having gotten enitty mention !', datetime.now() - start_time)

    elif k_entity == 0:  # ner
        model = Model(device=device, component='ner', model_path_load='parameter/ner.model',
                      model_path_save='parameter/ner.model')
        print('load model ...')
        model.load()
        print('loading data ...')
        data = DataLoader(
            NERDataset(data=data, device=device, is_test=True, is_train=False),
            batch_size=1, num_workers=0, collate_fn=test_collate)

        new_data = []
        print('filting entity ...')
        for _, item in enumerate(data):
            tags = model.predict(item)
            # print(tags, item['mention'], item['question'], item['label'])
            new_data.append({
                'question': item['origin_data']['question'],
                'tokens': get_mentions(item['question'], tags[0]),
                'origin_data': item['origin_data']
            })
        data = new_data
        json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print('having gotten NER !', datetime.now() - start_time)

        print('\ngetting entity candidates ...')
        data = get_entity_candidates(data)
        for node in data:
            node['entity_mention'] = node['entity_candidates']
        json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print('having gotten enitty candidates !', datetime.now() - start_time)

    ##############
    else:
        print('\ngetting entity candidates ...')
        data = get_entity_candidates(data)
        json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print('having gotten enitty candidates !', datetime.now() - start_time)

        model = Model(device=device, component='entity', model_path_load='parameter/entity.' + str(k_entity) + '.model',
                      model_path_save='parameter/entity.' + str(k_entity) + '.model', k=k_entity)
        print('load model ...')
        model.load()
        print('loading data ...')
        data = DataLoader(
            EntityDataset(data=data, device=device, is_test=True, is_train=False),
            batch_size=1, num_workers=0, collate_fn=test_collate)

        new_data = []
        print('filting entity ...')
        for i, item in enumerate(data):
            index = model.predict(item)
            new_data.append({
                'question': item['origin_data']['question'],
                'entity_mention': get_topk_entity(index, item),  # dict: key value
                'origin_data': item['origin_data']
            })
        data = new_data
        print('filter entity has finished !', datetime.now() - start_time)
        json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    pair  ##################################
    '''
    print('\ngetting entity-relation pair ...')
    data = get_entity_relation_pair(data)
    json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    model = Model(device=device, component='pair', model_path_load='parameter/pair.' + str(k_pair) + '.model',
                  model_path_save='parameter/pair.' + str(k_pair) + '.model', k=k_pair)
    print('load model ...')
    model.load()
    print('loading data ...')
    data = DataLoader(
        PairDataset(data=data, device=device, is_test=True, is_train=False),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    new_data = []
    print('filting entity relation pair ...')
    for i, item in enumerate(data):
        index = model.predict(item)
        pair = get_topk_pair(index, item)  # dict: key value
        new_data.append({
            'question': item['origin_data']['question'],
            'pair': pair,  # e +/- r
            'origin_data': item['origin_data']['origin_data']
        })
    data = new_data
    print('filter pair has finished !', datetime.now() - start_time)
    json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    get query graph candidates #########
    '''
    print('getting query graph ... ')
    data = get_query_graph(data)
    json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    rank ################################
    '''
    model = Model(device=device, component='qg', model_path_load='parameter/qg.model',
                  model_path_save='parameter/qg.model')
    print('load model ...')
    model.load()
    print('loading data ...')
    data = DataLoader(
        QGDataset(data=data, device=device, is_test=True, is_train=False),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    print('ranking ...')
    new_data = []
    for _, item in enumerate(data):
        index = model.predict(item)
        pred = item['qg_'][index]
        new_data.append({
            'question': item['origin_data']['question'],
            'pred': pred,  # query graph
            'origin_data': item['origin_data']['origin_data']
        })
    data = new_data
    print('predicted all query graph ', datetime.now() - start_time)
    json.dump(data, open(path + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    return data


def test_result(path):
    start_time = datetime.now()

    jieba.load_userdict(OUR_JIEBA_DICT)
    print('loaded user dict: ', datetime.now() - start_time)
    seg = pkuseg.pkuseg()  # 默认初始化
    stopwords = json.load(open(STOPWORDS_PATH, 'r', encoding='utf-8'))

    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            question = re.findall(r'q[0-9]+:(.+)', line.rstrip('？\n').rstrip('?\n'))[0]
            splited_question = [each for each in jieba.lcut(question, cut_all=True)
                                if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
            splited_question += [each for each in seg.cut(question)
                                 if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
            # "XXXX" 大概率是entity
            for each in re.findall(r'"(.+)"', question):
                splited_question.append(each)
            data.append({
                'question': question,
                'tokens': [each for each in list(set(splited_question)) if each not in stopwords]  # 去重
            })
    print('loaded data ', datetime.now() - start_time)

    data = result(path, data)

    '''
    get answer  #############################
    '''
    # data = json.load(open(path + '.json', 'r', encoding='utf-8'))

    print('getting answer ...')
    with open(path + '.' + str(k_entity) + '.' + str(k_pair) + '.answer.txt', 'w', encoding='utf-8') as answer_file:
        count = 0
        for node in data:
            # 塑造sparql
            count += 1
            query = build_sparql(node['pred'])
            answer = get_answer(query)
            answer_file.write('\t'.join(answer) + '\n')

            if count % 100 == 0:
                print(count, ' finished')
                # break
    print('got answers ', datetime.now() - start_time)


def dev_result(path):
    start_time = datetime.now()

    jieba.load_userdict(OUR_JIEBA_DICT)
    print('loaded user dict: ', datetime.now() - start_time)
    seg = pkuseg.pkuseg()  # 默认初始化
    stopwords = json.load(open(STOPWORDS_PATH, 'r', encoding='utf-8'))

    data = json.load(open(path, 'r', encoding='utf-8'))

    for node in data:
        question = node['question']
        splited_question = [each for each in jieba.lcut(question, cut_all=True)
                            if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
        splited_question += [each for each in seg.cut(question)
                             if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
        # "XXXX" 大概率是entity
        for each in re.findall(r'"(.+)"', question):
            splited_question.append(each)
        node['tokens'] = [each for each in list(set(splited_question)) if each not in stopwords]

    new_data = result(path, data)
    # new_data = json.load(open(path + '.json', 'r', encoding='utf-8'))

    count = 0
    precision = 0
    recall = 0
    f1 = 0
    # error = []
    for node in new_data:
        # 塑造sparql
        count += 1
        query = build_sparql(node['pred'])

        answer = get_answer(query)
        node['pred_answer'] = answer
        res = f1_score(node['origin_data']['answer'], answer)

        precision += res[0]
        recall += res[1]
        f1 += res[2]

        if count % 100 == 0:
            print(count, ' finished')
            # break
    precision /= len(new_data)
    recall /= len(new_data)
    f1 /= len(new_data)
    print('准确率： %f, 召回率：%f, F1-score：%f' % (precision, recall, f1))
    # json.dump(error, open('data/parser_answer_error.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(new_data, open('data/'+str(k_entity)+'.'+str(k_pair)+'.dev.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(datetime.now() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpu, epochs, batch_size, lr, is_train')
    parser.add_argument('--k_entity', '-entity', help='entity', default=5, type=int)
    parser.add_argument('--k_pair', '-pair', help='pair', default=5, type=int)
    parser.add_argument('--is_test', '-test', help='是否为测试', default=False, type=ast.literal_eval)
    parser.add_argument('--gpu', '-gpu', help='选择显卡', default=0, type=int)
    args = parser.parse_args()

    k_entity = args.k_entity
    k_pair = args.k_pair
    gpu = args.gpu

    if args.is_test:
        test_result(TEST_DATA_PATH)
    else:
        dev_result(FILTER_DEV_DATA_PATH)
