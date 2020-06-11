**CCKS2020评测任务一：新冠知识图谱构建与问答（四）新冠知识图谱问答评测**

****

#### requirements：

1. jieba-0.42.1
2. pytorch-1.2.0



#### 文件说明

fileTree.txt呈现项目的文件目录树，所需data和resources可从CCKS2020评测下载

1. clean_pkubase.py 清除pkubase中不合理的triple, 并存储字典

2. get_mention2ent.py 从pkubase和mention2ent.txt中得到mention2ent.json, 以及jieba分词需要的自定义词表。

3. train_filter_prepare.py 读取训练数据，并对question、sparql和answer进行处理，其中用到jieba进行了分词（cut_all=True）。得到所有可能的实体候选，需要后续过滤。

