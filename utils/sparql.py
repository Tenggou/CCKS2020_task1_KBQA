import re
import numpy as np

from utils.SPARQLWrapper import JSON, SPARQLWrapper

from utils.configure import PKUBASE_ENDPOINT, QUERY_GRAPH


def sparql_query(query):
    sparql = SPARQLWrapper(PKUBASE_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    return sparql.query().convert()


def sparql_parser(line):
    '''
    # !todo ttl长度为2暂时不考虑(即filter语句)
    20200516 解析sparql，得到entity literal value及其到答案节点的路径
    '''
    # 获得答案节点
    goal = re.findall(r'select[ |a-z]*(\?.+?) where .+', line.lower())[0]

    ttls = list(filter(lambda x: not re.fullmatch(r' *', x), re.findall(r'.*{(.+)}', line)[0].split('. ')))
    ttls = [list(filter(lambda x: x, re.findall(r'\?.|<.+?>|".+?"', ttl))) for ttl in ttls]
    ttls = [ttl for ttl in ttls if len(ttl) == 3]

    var = []  # 待删除的变量
    # 解析包含entity literal value的ttls，将SPARQL切分成实体到答案变量的查询路径
    path = []
    for ttl in ttls:
        if '?' not in ttl[0] and '?' not in ttl[-1]:  # 过滤 e1 ?v e2
            continue

        # 从goal出发搜索path
        if ttl[0] == goal:
            path.append([ttl[-1], '-', ttl[1], goal])  # 前一个变量 关系方向 关系 答案变量
            var.append(goal)
        elif ttl[-1] == goal:
            path.append([ttl[0], '+', ttl[1], goal])
            var.append(goal)

    flag = 1  # 有无新ttl加入
    while flag:
        flag = 0
        new_path = []
        for p in path:  # p为查询路径
            if re.fullmatch('<.+?>|".+?"', p[0]):  # 到达实体，无需再添加
                new_path.append(p)
                continue
            temp = []
            for ttl in ttls:
                if ttl[0] == p[0] and ttl[-1] not in p:
                    temp.append([ttl[-1], '-', ttl[1]])  # 前一个变量 关系方向 关系
                elif ttl[-1] == p[0] and ttl[0] not in p:
                    temp.append([ttl[0], '+', ttl[1]])
            if temp:
                for t in temp:  # 一个节点存在多个关系
                    var.append(p[0])
                    new_path.append(t+p)
                    flag = 1
            else:
                new_path.append(p)
        path = new_path

    parse = [[e for e in p if e not in var] for p in path]
    for p in parse:
        if len(p) % 2 != 1:
            print(line)
            print(parse)
            break
    return parse, line.strip()  # path, sparql


def build_sparql(parse):
    # 根据之前解析的结果重新组建成SPARQL
    sparql = 'select distinct ?x from ' + QUERY_GRAPH + ' where {{{query}}}'
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
