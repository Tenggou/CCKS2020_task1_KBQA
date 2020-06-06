import json
from datetime import datetime

'''
pkubase: 
1. 清理virtuoso无法加载的ttl
2. 收集所有字符
'''
start_time = datetime.now()
characters = []
count = 0
with open('resources/pkubase-complete-2020/pkubase-complete.txt', 'r', encoding='utf-8') as base:
    with open('resources/pkubase-complete-2020/pkubase-clean.ttl', 'w', encoding='utf-8') as new_base:
        for line in base:
            elements = line.rstrip(' .\n').split('\t')
            characters += list(line.rstrip(' .\n').replace('\t', ''))
            count += 1
            if count % 100000 == 0:
                # 过滤重复字符，避免内存不足
                characters = list(set(characters))
                print(count, ' finished, ', datetime.now()-start_time)
                # break
            # if len(elements) != 3 or len(re.findall(r'<|"(.+?)>|"', line)) != 3:
            # 下列字符使virtuoso加载语法错误
            if len(elements) != 3 or any(['<' in e[1:-1] for e in elements]) or any(['>' in e[1:-1] for e in elements]) \
                    or any(['"' in e[1:-1] for e in elements]) or any(['\\' in e[1:-1] for e in elements]) \
                    or any([not (e[0] == '<' or e[0] == '"') or not (e[-1] == '>' or e[-1] == '"') for e in elements]):
                # print(line)
                continue
            new_base.write(line)  # 逐行写
# 过滤重复字符
characters = list(set(characters))
json.dump(characters, open('resources/characters.txt', 'w', encoding='utf-8'), ensure_ascii=False)
print('all finished ', datetime.now() - start_time, ' ', count)
# 66499746（complete.txt，3.37G） 66476659（clean.ttl，3.43G）


# count = 0
# with open('resources/pkubase-complete-2020/pkubase-clean.ttl', 'r', encoding='utf-8') as base:
# # with open('resources/pkubase-complete-2020/pkubase-complete.txt', 'r', encoding='utf-8') as base:
#     for line in base:
#         count += 1
#         # if count >= 66227924:  # 66205514
#         #     print(line)
#         # if count == 66227924:
#         #     break
# print(count)

