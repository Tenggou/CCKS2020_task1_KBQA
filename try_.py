import json
import random
import re
from datetime import datetime

import torch
from SPARQLWrapper import SPARQLWrapper, JSON


def check_query_grap():
    # 查询图形状
    templates = {}
    data = json.load(open('data/train_filter/train.json', 'r', encoding='utf-8')) + \
           json.load(open('data/train_filter/dev.json', 'r', encoding='utf-8'))
    print(len(data))
    for node in data:
        template = []
        for p in node['parse']:
            t = ''
            for e in p:
                if e == '+' or e == '-':
                    t += e
            template.append(t)
        template = sorted(template, key=lambda x: len(x))
        template = '_'.join(template)
        if template in templates:
            templates[template] += 1
        else:
            templates[template] = 1
    templates = sorted(templates.items(), key=lambda x: x[1])
    print(json.dumps(templates, indent=4))


if __name__ == '__main__':
    # check_query_grap()

    ll = '{} {} ?x . {} {} ?x . '
    lr = '{} {} ?x . ?x {} {} . '
    rl = '?x {} {} . {} {} ?x . '
    rr = '?x {} {} . ?x {} {} . '
    d = {
            "脐带打结": [
                [
                    "<脐带打结>",
                    "-",
                    "<涉及疾病>"
                ],
                [
                    "<脐带打结>",
                    "+",
                    "<openkg_uri>"
                ],
                [
                    "<脐带打结>",
                    "+",
                    "<涉及检查>"
                ],
                [
                    "<脐带缠绕>",
                    "+",
                    "<疾病病因>"
                ]
            ],
            "肺栓塞": [
                [
                    "<肺栓塞>",
                    "-",
                    "<涉及疾病>"
                ],
                [
                    "<肺栓塞>",
                    "-",
                    "<相关疾病>"
                ]
            ],
            "脐带": [
                [
                    "<脐带_（人体结构）>",
                    "+",
                    "<作用>"
                ]
            ],
            "应该": [
                [
                    "<只为爱>",
                    "-",
                    "<代表作品>"
                ],
                [
                    "\"应该\"",
                    "-",
                    "<中文名称>"
                ],
                [
                    "<应该_（网络小说）>",
                    "-",
                    "<登场作品>"
                ]
            ]
        }
    pair = list(d.values())
    for i in range(len(pair)):
        for pre in pair[i]:
            print(range(i+1, len(pair)))
            for j in range(i+1, len(pair)):
                print(j)
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
                    print(query)
