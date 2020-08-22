

**CCKS2020评测任务一：新冠知识图谱构建与问答（四）新冠知识图谱问答评测**

****
#### 后续尝试提升点

1. ~~试一下从在线endpoint获取答案，会不会优于自己的endpoint~~
2. ~~pkubase版本是否不一致？~~（官方给了新的pkubase）
3. 改进获取实体候选的方法
4. sparql parser改进，如filter regex
5. 选取最优的K值

#### 运行说明

##### 配置：

本实验基于python3(Anaconda3)，另外需要以下工具包：

1. pytorch-1.2.0
2. jieba-0.42.1
3. pkuseg

其他资源配置：

1. BERT中文词表和config文件已经保留，BERT中文模型参数需要下载。打开[链接](https://www.kaggle.com/soulmachine/pretrained-bert-models-for-pytorch?)，下载bert-base-chinese/pytorch_model.bin文件，并将其放至resource/bert-base-chinese目录下。
2. 下载知识图谱相关资源（链接：https://pan.baidu.com/s/1mQXgm7BtdQin3oLUoPqNXQ  提取码：cs35），将pkubase-complete2.txt和pkubase-mention2ent.txt放至resource目录下。
3. 下载数据集（链接：https://pan.baidu.com/s/1IqCfmkmAQTBz5K37Sv-7Kw 提取码：81ek），将task1-4_train_2020.txt和task1-4_valid_2020.questions放至data目录下。

##### 运行：

配置完运行环境后，开始执行程序。

1. 执行以下代码清理PKUBASE与收集mention2entity映射文件。约耗时1小时。

   ```shell
   bash resource_prepare.sh
   ```

2. 使用virtuoso搭建PKUBASE的SPARQL查询端口（[搭建流程](https://blog.csdn.net/wtgwtg_/article/details/107963602)），搭建完成后修改utils/configure.py中的endpoint。

3. 准备数据与训练（使用RTX2080TI）。约耗时1+5+2+7小时。‘

   ```shell
   bash data_prepare_and_train_modules.sh
   ```

4. 获取验证集和测试集的答案。约耗时2小时。

   ```
   python test.py
   ```

#### 训练记录

**验证提交阶段：**

验证集只发布了问题，不包含SPARQL，所以本文随机从训练集中划出十分之一（即400个问题）作为实际验证集，剩余3600个问题作为实际训练集。

| 数据集         | filter | rank   | precision | recall | F1     |
| -------------- | ------ | ------ | --------- | ------ | ------ |
| 实际验证集     | 0.8931 | 0.7268 | 0.8177    | 0.8325 | 0.8179 |
| 官方发布验证集 |        |        | 0.8051    | 0.8288 | 0.8073 |

|                     | 官方发布训练集         |
| ------------------- | ---------------------- |
| 平均实体候选数      | 38.89                  |
| 含正确实体比例      | 0.9384                 |
| 实体-关系对平均数量 | 652.31                 |
| 含正确实体关系对率  | 0.9334                 |
| 查询图平均数量      | 330+                   |
| 含正确查询图率      | 0.8820（0.885，0.855） |

------

**测试提交阶段：**

| 数据集         | filter | rank | precision | recall | F1   |
| -------------- | ------ | ---- | --------- | ------ | ---- |
| 官方发布验证集 |        |      |           |        |      |
| 官方发布测试集 |        |      |           |        |      |

|                     | 官方发布训练集 | 官方发布验证集 |
| ------------------- | -------------- | -------------- |
| 平均实体候选数      |                |                |
| 含正确实体比例      | 0.938          |                |
| 实体-关系对平均数量 |                |                |
| 含正确实体关系对率  |                |                |

#### 数据集统计

| **问题类型** | **训练集** | **验证集** | **测试集** |
| ------------ | ---------- | ---------- | ---------- |
| 单实体单跳   | 2324       |            |            |
| 单实体多跳   | 738        |            |            |
| 多实体单跳   | 777        |            |            |
| 其他         | 161        |            |            |
| 总数         | 4000       | 1529       |            |

