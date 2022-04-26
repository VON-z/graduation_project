# -*- encoding: utf-8 -*-
'''
@File    :   algorithm.py
@Time    :   2022/03/27 13:33:46
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   Implement some algorithms to solve the problem.
'''

# here put the standard library
import random
import math

# here put the third-party packages

# here put the local import source
from multiplex_network.group import Group
# from multiplex_network.network import Network


class Algorithom():
    """Implement different algorithms.
    Including grouping results and total payoff.
    """
    def __init__(self, network) -> None:
        """Initialization.
        """
        self.network = network
        self.groups = []
        self.total_payoff = 0

    def cal_payoff(self):
        """Calculate total payoff.
        """
        for g in self.groups:
            g.cal_payoff()
            self.total_payoff += g.payoff

    def write2file(self, path):
        """将分组结果写入磁盘文件

        Args:
            path (_type_): _description_
        """
        pass

    def random(self):
        """Randomly assign agents.
        """
        # Shuffle the set of agents
        agents = self.network.agents.copy()
        random.shuffle(agents)

        group_num = math.ceil(self.network.scale / Group.max_member_num) # the number of groups
        for idx in range(group_num):
            self.groups.append(Group(idx))
        for i in range(self.network.scale):
            self.groups[i % group_num].add_agent(agents[i])
        