

**CCKS2020评测任务一：新冠知识图谱构建与问答（四）新冠知识图谱问答评测**

****
#### requirements：

需要以下工具包：

1. anaconda3
2. jieba-0.42.1
3. pkuseg
4. pytorch-1.2.0
5. SPARQLWrapper

#### 运行说明

##### 训练：

训练前需要配置文件：从CCKS2020评测下载数据集和pkubase，从https://www.kaggle.com/soulmachine/pretrained-bert-models-for-pytorch/data下载bert-base-chinese的预训练参数和词表。其中数据集已经放入data\ccks_2020_7_4_Data，其余文件配置参考.gitignore。

1. python clean_pkubase.py 
2. python get_mention2ent.py 
3. python train_filter_prepare.py 
4. python train_filter.py
5. python train_rank_prepare.py
6. python train_rank.py

##### 测试：

训练后，执行python sparql.py

#### 文件说明

fileTree.txt呈现项目的文件目录树，所需data和resources可从CCKS2020评测下载

1. clean_pkubase.py 
   1. 清除pkubase中不合理的triple（virtuoso加载报错的TTL）
   2. 将pkubase中的字存为字典
2. get_mention2ent.py 
   1. 从pkubase和mention2ent.txt中得到mention2ent.json,
   2. 将mention2ent.json中的mention存为jieba分词需要的自定义词表。
3. train_filter_prepare.py 
   1. 读取训练数据
   2. 对sparql进行解析
   3. 用到jieba（cut_all=True）和pkuseg对问题进行分词。得到所有可能的实体候选。
   4. 根据实体获得实体-关系对候选。
4. train_filter.py 训练filter用于过滤实体-关系对
5. train_rank_prepare.py 使用过滤后的实体-关系对构建查询图候选
6. train_rank.py 训练rank
7. sparql.py 对验证集或测试集进行测试（输出答案）



#### 训练记录

因为目前（2020年06月28日）验证集只发布了问题，不包含SPARQL，所以本文随机从训练集中划出十分之一（即400个问题）作为实际验证集，剩余3600个问题作为实际训练集。

| 数据集         | 模型 | filter        | rank            | precision | recall | F1     |
| -------------- | ---- | ------------- | --------------- | --------- | ------ | ------ |
| 实际验证集     | LSTM | 83.30%/90.79% | 56.25% / 74.80% | 61.87%    | 64.00% | 61.90% |
| 实际验证集     | BERT | 83.33%/86.46% | 68.50% / 78.20% | 76.39%    | 78.68% | 76.52% |
| 官方发布验证集 | LSTM | -             | -               | 66.16%    | 68.24% | 66.26% |
| 官方发布验证集 | BERT | -             | -               | 80.85%    | 83.08% | 81.16% |

上表需要注意filter和rank的实际验证集是不一样的，而且LSTM只用jieba(cut_all)分词方法。

| 官方发布训练集      | jieba(cut_all) | +pkuseg | +date/ +" "/ + not in mention2txt/ +stopwords |
| ------------------- | -------------- | ------- | --------------------------------------------- |
| 平均实体候选数      | 45.16          | 52.88   | 38.91                                         |
| 含正确实体比例      | 92.73%         | 93.35%  | 93.82%                                        |
| 实体-关系对平均数量 | 928.45         | -       | 652.10                                        |
| 含正确实体关系对率  | 88.68%         | -       | 89.81%                                        |

#### 数据集统计

| **问题类型** | **训练集** | **验证集** | **测试集** |
| ------------ | ---------- | ---------- | ---------- |
| 单实体单跳   | 2324       | -          | -          |
| 单实体多跳   | 738        | -          | -          |
| 多实体单跳   | 777        | -          | -          |
| 其他         | 161        | -          | -          |
| 总数         | 4000       | 1529       | -          |

