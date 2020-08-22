import sys
import os
sys.path.append(os.path.abspath(''))

from datetime import datetime

from utils.configure import PKUBASE_COMPLETE_PATH, PKUBASE_CLEAN_PATH


if __name__ == '__main__':
    '''
    pkubase: 
    清理virtuoso无法加载的ttl
    '''
    print('clean_pkubase.py running')
    start_time = datetime.now()
    count = 0
    with open(PKUBASE_COMPLETE_PATH, 'r', encoding='utf-8') as base:
        with open(PKUBASE_CLEAN_PATH, 'w', encoding='utf-8') as new_base:
            for line in base:
                count += 1
                elements = line.rstrip(' .\n').split('\t')
                # 下列字符使virtuoso加载语法错误
                if len(elements) != 3 or any(['<' in e[1:-1] for e in elements]) or any(['>' in e[1:-1] for e in elements]) \
                        or any(['"' in e[1:-1] for e in elements]) or any(['\\' in e[1:-1] for e in elements]) \
                        or any([not (e[0] == '<' or e[0] == '"') or not (e[-1] == '>' or e[-1] == '"') for e in elements]):
                    # print(line)
                    continue
                new_base.write(line)  # 逐行写
                if count % 1000000 == 0:
                    print(f'{count} finished ', datetime.now() - start_time)
    print('clean_pkubase.py finished ', datetime.now() - start_time, ' the num of lines are ', count)
