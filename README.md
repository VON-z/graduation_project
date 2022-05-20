# graduation_project
网络结构化多智能体协作学习群组形成研究。

## multiplex_network

实现双层网络模型。

### agent.py

单智能体。包含：技能、学习效率。

### network.py

双层网络模型。包含智能体集合、两层网络权值、计算收益的辅助矩阵T1和T2。

## ga.py

实现遗传算法。

## sa.py

实现模拟退火算法。

## ukb.py

实现基于均一k均值聚类的分组算法。

## data.py

1.生成模拟数据。
2.保存/读取实验数据的函数。
3.将数据组织成表格。

## draw.py

1.绘制SA算法和GA算法的收敛曲线。
2.绘制SA算法、GA算法和UKB算法在不同网络下的收益条形图。

## main.py

运行实验，三种算法在各种参数各种网络模型下的完整实验。

## environment.yml

实验的python环境。可用anaconda安装实验环境。

```conda env create -f environment.yml```

## 实验结果/数据

由于上传文件大小限制50M，因此压缩包不包含下面这些内容，完整内容见 https://github.com/VON-z/graduation_project

### data文件夹

生成的网络连边数据和智能体集合数据。
运行data.py的Generate data.部分。

### result文件夹

三种算法在各种参数各种网络模型下的完整实验结果，具体对应的参数已用文件路径+文件名的方式标记。
完整运行main.py，产生完整的实验数据。

### table文件夹

将上述result文件夹中的所有实验数据整理成的表格。
运行data.py的Organize data into tables.部分。

### figure文件夹

0文件夹: 对应SA算法和GA算法的收敛曲线。
运行draw.py的Draw evaluation line chart.部分。

performance文件夹: 绘制的SA算法、GA算法和UKB算法在不同网络下的收益条形图。
运行draw.py的Draw performance bar.部分。