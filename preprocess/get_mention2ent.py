import sys
import os

import jieba
from pkuseg import pkuseg

sys.path.append(os.path.abspath(''))

import json
import re

from datetime import datetime

from utils.configure import PKUBASE_CLEAN_PATH, PKUBASE_MENTION2ENT_ORIGIN, PKUBASE_MENTION2ENT_COLLECTED, \
    OUR_JIEBA_DICT

if __name__ == '__main__':
    '''
    1. 从pkubase-mention2ent.txt（候选没有标记entity或是literal全部作为entity处理）和pkubase获取mention2ent.json
    2. 把所有可能的mention保存为分词表
    '''
    print('get_mention2ent.py running')
    start_time = datetime.now()
    mention2ent = {}
    count = 0
    seg = pkuseg()

    def get_mentions(entity):
        # 获取实体的mention，并对其进行分词
        return entity[1:-1].split('_')[0]

    # 读取pkubase中的所有entity和literal并对应它们的mention
    with open(PKUBASE_CLEAN_PATH, 'r', encoding='utf-8') as base:
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

            mention = get_mentions(elements[0])
            # 过滤包含空格的mention，因为后面的分词的效果（获取实体候选）下降了
            if ' ' not in mention:
                if mention in mention2ent:
                    mention2ent[mention].append(elements[0])
                else:
                    mention2ent[mention] = [elements[0]]

            mention = get_mentions(elements[-1])
            # 过滤包含空格的mention
            if ' ' not in mention:
                if mention in mention2ent:
                    mention2ent[mention].append(elements[-1])
                else:
                    mention2ent[mention] = [elements[-1]]

    # 读取比赛举办方提供的pkubase-mention2ent.txt，去重
    with open(PKUBASE_MENTION2ENT_ORIGIN, 'r', encoding='utf-8') as file:
        for line in file:
            temp = line.strip().split('\t')
            if len(temp) != 3:
                # print(temp)
                continue

            if count % 500000 == 0:
                for key, value in mention2ent.items():
                    mention2ent[key] = list(set(value))
                print(count, ' finished, ', datetime.now()-start_time)

            count += 1
            mention, entity, num = temp
            if ' ' in entity or ' ' in mention or not re.fullmatch('[0-9]+', temp[-1]):
                # 过滤不合法情况
                continue
            if mention in mention2ent:
                if '_' in entity:
                    mention2ent[mention].append('<'+entity+'>')
                else:
                    mention2ent[mention].append('<'+entity+'>')
                    mention2ent[mention].append('"'+entity+'"')
            else:
                if '_' in entity:
                    mention2ent[mention] = ['<'+entity+'>']
                else:
                    mention2ent[mention] = ['<'+entity+'>', '"'+entity+'"']

    # 最后一次未满500000，需要过滤
    for key, value in mention2ent.items():
        mention2ent[key] = list(set(value))

    # 得到结巴自定义词表
    with open(OUR_JIEBA_DICT, 'w', encoding='utf-8') as dictionary:
        dictionary.write(' 3 n\n'.join(list(mention2ent.keys())) + ' 3 n')

    json.dump(mention2ent, open(PKUBASE_MENTION2ENT_COLLECTED, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print('get_mention2ent.py finished ', datetime.now()-start_time, 'the num of mention2ent are ', count)
