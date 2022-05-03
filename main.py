# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/01/29 12:46:10
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   None
'''

# here put the standard library

# here put the third-party packages

# here put the local import source
from multiplex_network.agent import Agent
from multiplex_network.group import Group
from multiplex_network.network import Network
from ga import GA

# 超参数设置
SKILL_NUM = 5
MAX_MEMBER_NUM = 5
NETWORK_SCALE = 50

# 初始化Agent类变量
Agent.skill_num = SKILL_NUM

# 初始化Group类变量
Group.max_member_num = MAX_MEMBER_NUM

# 初始化Network类变量
Network.max_scale = NETWORK_SCALE

# 构建网络
network = Network(NETWORK_SCALE)
# 初始化Group类的邻接矩阵
Group.adjacency_matrix = network.adjacency_matrix
Group.motivation_network = network.motivation_network

# 实例化算法类
algorithm = Algorithom(network)

# 算法1:随机分组
algorithm.random()
# for i in range(10):
#     for j in range(5):
#         print(algorithm.groups[i].agents[j].index)
#     print('\n')
algorithm.cal_payoff()
print(algorithm.total_payoff)
