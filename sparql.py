import json
from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime


endpoint = 'http://10.201.180.179:8890/sparql'


def build_sparql(parse):
    # !todo ttl长度为2暂时不考虑(filter语句)
    sparql = 'select distinct ?x from <http://pkubase.cn> where {{{query}}} LIMIT 100'
    query = []
    answer = '?x'
    for p in range(len(parse)):
        last_v = parse[p]['entity']
        for d in range(len(parse[p]['direction'])):
            v = '?v' + str(p) + str(d)
            ttl = []
            if len(parse[p]['path'][d]) == 2:
                continue  # filter
            if parse[p]['direction'][d] == 'l':
                ttl = [last_v, parse[p]['path'][d][1], v]
            elif parse[p]['direction'][d] == 'r':
                ttl = [v, parse[p]['path'][d][1], last_v]
            query.append(' '.join(ttl))
            last_v = v
        query[-1] = query[-1].replace(last_v, answer)
    query.append(' ')
    # !todo 构建查询图时合并相同关系的ttl
    query = ' . '.join(query)
    return sparql.format(query=query)


def get_answer(query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    results = sparql.query().convert()
    answer = []
    for result in results['results']['bindings']:
        if 'x' in result:
            answer.append(result['x']['value'])
    return answer


def f1_score(true, pred):
    count = 0
    for t in true:
        if t[1:-1] in pred:
            count += 1
    # print(true)
    # print(pred)
    precision = count/len(pred) if len(pred) else 0  # 预测的准确率
    recall = count/len(true) if len(pred) else 0  # 召回率
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    return precision, recall, f1


if __name__ == '__main__':
    '''
    测试还原解析的sparql方法的效果
    '''
    start_time = datetime.now()
    path = 'data/train.json'
    data = json.load(open(path, 'r', encoding='utf-8'))
    count = 0
    precision = 0
    recall = 0
    f1 = 0
    for node in data:
        # 塑造sparql
        count += 1
        query = build_sparql(node['parse'])

        # print(query)

        answer = get_answer(query)
        result = f1_score(node['answer'], answer)
        precision += result[0]
        recall += result[1]
        f1 += result[2]

        if count % 100 == 0:
            print(count, ' finished')
        # print(count, ' finished')
    precision /= len(data)
    recall /= len(data)
    f1 /= len(data)
    # 准确率： 0.930954, 召回率：0.942679, F1-score：0.931585
    # 0:08:39.175901
    print('准确率： %f, 召回率：%f, F1-score：%f' % (precision, recall, f1))
    print(datetime.now() - start_time)
