import json
import re
from datetime import datetime

import jieba
import pkuseg
import torch
from torch.utils.data import DataLoader

from model.model import Model
from preprocess.train_filter_prepare import get_entity_candidates, get_entity_relation_pair
from preprocess.train_rank_prepare import get_query_graph
from utils.configure import FILTER_DEV_DATA_PATH, K_PAIR, DEV_RESULT_PATH, OUR_JIEBA_DICT, STOPWORDS_PATH, \
    TEST_DATA_PATH, RANK_DEV_DATA_PATH
from utils.dataset import test_collate, RankDataset, FilterDataset
from utils.evaluate import get_topk_candidate
from utils.sparql import sparql_query, build_sparql


def get_answer(query):
    results = sparql_query(query)
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


def dev_result():
    start_time = datetime.now()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检测GPU
    # data = json.load(open(FILTER_DEV_DATA_PATH, 'r', encoding='utf-8'))
    # '''
    # filter  ##################################
    # '''
    # if torch.cuda.is_available():
    #     from torch.backends import cudnn
    #     cudnn.benchmark = True
    # print('device name is ', device)
    #
    # model = Model(device=device, component='filter', model_path='parameter/filter.' + str(K_PAIR) + '.model')
    # print('load model ...')
    # model.load()
    # print('loading data ...')
    # data = DataLoader(
    #     FilterDataset(data=data, device=device, is_test=True, is_train=False),
    #     batch_size=1, num_workers=0, collate_fn=test_collate)
    #
    # new_data = []
    # print('filting entity relation pair ...')
    # for i, item in enumerate(data):
    #     index = model.predict_filter(item)
    #     pair = get_topk_candidate(index, item)  # dict: key value
    #     new_data.append({
    #         'question': item['origin_data']['question'],
    #         'pair': pair,  # e +/- r
    #         'answer': item['origin_data']['answer'],
    #         'sparql': item['origin_data']['sparql'],
    #         'parse': item['origin_data']['parse']
    #     })
    # data = new_data
    # print('filter pair has finished !', datetime.now() - start_time)
    #
    # '''
    # get query graph candidates #########
    # '''
    # print('getting query graph ... ')
    # data = get_query_graph(data)

    '''
    rank ################################
    '''
    data = json.load(open(RANK_DEV_DATA_PATH, 'r', encoding='utf-8'))

    model = Model(device=device, component='rank', model_path='parameter/rank.model')
    print('load model ...')
    model.load()
    print('loading data ...')
    data = DataLoader(
        RankDataset(data=data, device=device, is_test=True, is_train=False),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    print('ranking ...')
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
    data = new_data
    print('predicted all query graph ', datetime.now() - start_time)

    '''
    evaluating ################################
    '''
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
        result = f1_score(node['answer'], answer)

        precision += result[0]
        recall += result[1]
        f1 += result[2]

        if count % 100 == 0:
            print(count, ' finished')
            # break
    precision /= len(data)
    recall /= len(data)
    f1 /= len(data)
    print('准确率： %f, 召回率：%f, F1-score：%f' % (precision, recall, f1))
    json.dump(new_data, open(DEV_RESULT_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(datetime.now() - start_time)


def test_result(path):
    """
    完整的测试流程：从问题到查询图
    :param path:
    :return:
    """
    start_time = datetime.now()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检测GPU

    jieba.load_userdict(OUR_JIEBA_DICT)
    print('loaded user dict: ', datetime.now() - start_time)
    seg = pkuseg.pkuseg()  # 默认初始化
    stopwords = json.load(open(STOPWORDS_PATH, 'r', encoding='utf-8'))

    '''
    准备测试数据  ##################################
    '''
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            question = re.findall(r'q[0-9]+:(.+)', line.rstrip('？\n').rstrip('?\n'))[0]

            # 分词
            splited_question = [each for each in jieba.lcut(question, cut_all=True)
                                if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
            splited_question += [each for each in seg.cut(question)
                                 if each not in ['', ' ', '，', ',', '。', '.', '?', '？', '"', '\'']]
            for each in re.findall(r'"(.+)"', question):
                if each not in splited_question:
                    splited_question.append(each)
            splited_question = [each for each in splited_question if each not in stopwords]
            splited_question = list(set(splited_question))

            data.append({
                'question': question,
                'tokens': splited_question,
                'entity_candidates': {}
            })

    print('\ngetting entity candidates ...')
    data = get_entity_candidates(data)
    json.dump(data, open(path+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    print('\ngetting entity-relation pair ...')
    data = get_entity_relation_pair(data)
    json.dump(data, open(path+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

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
    print('loading data ...')
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
            'pair': pair  # e +/- r
        })
    data = new_data
    print('filter pair has finished !', datetime.now() - start_time)
    json.dump(data, open(path+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    get query graph candidates #########
    '''
    print('getting query graph ... ')
    data = get_query_graph(data)
    json.dump(data, open(path+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    rank ################################
    '''
    model = Model(device=device, component='rank', model_path='parameter/rank.model')
    print('load model ...')
    model.load()
    print('loading data ...')
    data = DataLoader(
        RankDataset(data=data, device=device, is_test=True, is_train=False),
        batch_size=1, num_workers=0, collate_fn=test_collate)

    print('ranking ...')
    new_data = []
    for _, item in enumerate(data):
        index = model.predict_rank(item)
        pred = item['qg_'][index]
        new_data.append({
            'question': item['origin_data']['question'],
            'pred': pred  # query graph
        })
    data = new_data
    print('predicted all query graph ', datetime.now() - start_time)
    json.dump(data, open(path+'.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    '''
    get answer  #############################
    '''
    print('getting answer ...')
    with open(path+'.answer.txt', 'w', encoding='utf-8') as answer_file:
        count = 0
        for node in data:
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
    dev_result()

    test_result(TEST_DATA_PATH)
