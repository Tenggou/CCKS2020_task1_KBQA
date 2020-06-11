import json
import re

from datetime import datetime


if __name__ == '__main__':
    '''
    1. 从pkubase-mention2ent.txt（候选没有标记entity或是literal全部作为entity处理）和pkubase获取mention2ent.json
    2. 把所有mention保存为分词表
    '''
    start_time = datetime.now()
    mention2ent = {}
    count = 0

    def get_mention(entity):
        return entity[1:-1].split('_')[0], entity[1:-1]

    # 读取pkubase中的所有entity和literal并对应它们的mention
    # with open('resources/pkubase-complete-2020/pkubase-complete.txt', 'r', encoding='utf-8') as base:
    with open('resources/pkubase-complete-2020/pkubase-clean.ttl', 'r', encoding='utf-8') as base:
        for line in base:
            count += 1
            if count % 500000 == 0:
                # 过滤重复项，加快append
                for key, value in mention2ent.items():
                    mention2ent[key] = list(set(value))
                print(count, ' finished, ', datetime.now()-start_time)

            elements = line.rstrip(' .\n').split('\t')
            if len(elements) != 3:
                continue

            mention, name = get_mention(elements[0])
            # 过滤包含空格的mention，后面的分词的效果（实体候选）下降了
            if ' ' not in mention:
                if mention in mention2ent:
                    mention2ent[mention].append(elements[0])
                else:
                    mention2ent[mention] = [elements[0]]

            mention, name = get_mention(elements[-1])
            # 过滤包含空格的mention
            if ' ' not in mention:
                if mention in mention2ent:
                    mention2ent[mention].append(elements[-1])
                else:
                    mention2ent[mention] = [elements[-1]]

    # 读取比赛举办方提供的pkubase-mention2ent.txt，去重
    with open('resources/pkubase-mention2ent.txt', 'r', encoding='utf-8') as file:
        for line in file:
            temp = line.strip().split('\t')

            count += 1

            if len(temp) != 3:
                # print(temp)
                continue

            if count % 500000 == 0:
                for key, value in mention2ent.items():
                    mention2ent[key] = list(set(value))
                print(count, ' finished, ', datetime.now()-start_time)

            mention, entity, num = temp

            if ' ' in entity or ' ' in mention or not re.fullmatch('[0-9]+', temp[-1]):
                # 过滤不合法情况
                continue

            if mention in mention2ent:
                if '<'+entity+'>' not in mention2ent[mention]:
                    mention2ent[mention].append('<'+entity+'>')
            else:
                mention2ent[mention] = ['<'+entity+'>']

    for key, value in mention2ent.items():
        mention2ent[key] = list(set(value))

    # 得到结巴自定义词表
    with open('resources/dict.txt', 'w', encoding='utf-8') as dictionary:
        dictionary.write(' 3 n\n'.join(list(mention2ent.keys())) + ' 3 n')

    json.dump(mention2ent, open('resources/mention2ent.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    print('all finished ', datetime.now()-start_time, ' ', count)  # 80406617
