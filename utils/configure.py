PKUBASE_COMPLETE_PATH = 'resource/pkubase-complete2.txt'  # 原PKUBASE
PKUBASE_CLEAN_PATH = 'resource/pkubase-clean.ttl'  # 清理后的PKUBASE，用于virtuoso搭建查询端口（实际使用）

PKUBASE_MENTION2ENT_ORIGIN = 'resource/pkubase-mention2ent.txt'  # 原实体提及到实体候选的映射文件
PKUBASE_MENTION2ENT_COLLECTED = 'resource/pkubase-mention2ent-collected.json'  # 额外收集的实体提及到实体候选的映射文件（实际使用）

# 数据集
TRAIN_DATA_PATH = 'data/task1-4_train_2020.txt'  # 官方训练数据集
DEV_DATA_PATH = ''
TEST_DATA_PATH = 'data/task1-4_valid_2020.questions'  # 官方验证数据集（验证提交，只有问题

FILTER_TRAIN_DATA_PATH = 'data/filter/train.json'  # 验证提交，实际filter训练集
FILTER_DEV_DATA_PATH = 'data/filter/dev.json'  # 验证提交，实际filter验证集
RANK_TRAIN_DATA_PATH = 'data/qg/train.json'  # 验证提交，实际query graph rank训练集
RANK_DEV_DATA_PATH = 'data/qg/dev.json'  # 验证提交，实际query graph rank验证集

# 错误分析
ENTITY_ERROR_PATH = 'data/entity_error.json'  # 预处理过程中，未能获得正确的entity的问题
DEV_RESULT_PATH = 'data/our_dev_result.json'  # 实际验证集最终答案

# 资源
STOPWORDS_PATH = 'resource/stopwords.txt'  # 停用词
OUR_JIEBA_DICT = 'resource/dict.txt'  # 收集得到的用于结巴分词的自定义词表
BERT_BASE_CHINESE = 'resource/bert-base-chinese'  # BERT中文模型参数
BERT_VOCAB = 'resource/bert-base-chinese-vocab.txt'  # BERT中文词表

# SPARQL相关
PKUBASE_ENDPOINT = 'http://localhost:8890/sparql'  # SPARQL查询端口
QUERY_GRAPH = '<http://pkubase.cn>'