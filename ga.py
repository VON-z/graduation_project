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

# here put the third-party packages
import scipy.linalg

# here put the local import source
from numpy import np
from multiplex_network.group import Group
# from multiplex_network.network import Network


class GA():
    """Implement different algorithms.
    Including grouping results and total payoff.
    """
    def __init__(self, network, group_scale, P, pc, pm, it) -> None:
        """Initialization.

        Args:
            network (Network): network model.
            group_scale (int): max member number in a group.
            P (int): population size.
            pc (float): probability of performing crossover.
            pm (float): probability of mutation.
            it (int): iteration rounds.
        """
        self.network = network
        self.group_scale = group_scale
        self.P = P
        self.pc = pc
        self.pm = pm
        self.it = it

        self.block_diagonal_matrix = None
        self.generate_block_diagonal_matrix()

        self.grouping_matrix = None # The tensor representing a series of grouping results。
        self.total_payoff = None   # Total payoff of a series of grouping results.

    def generate_block_diagonal_matrix(self):
        """Generate block diagonal matrix.
        """
        # Calculate how many groups are missing an agent and how many groups are complete.
        group_num = self.network.scale // self.group_scale + \
            self.network.scale % self.group_scale
        missing = group_num * self.group_scale - self.network.scale
        complete = group_num - missing

        # Generate block diagonal matrix.
        blocks = []
        for _ in range(complete):
            blocks.append(np.ones([self.group_scale, self.group_scale], dtype=int))
        for _ in range(missing):
            blocks.append(np.ones[self.group_scale-1, self.group_scale-1], dtype=int)
        self.block_diagonal_matrix = scipy.linalg.block_diag(*blocks)


    def generate_initial_population(self):
        """Generate initial population.
        """
        T = np.zeros([self.network.scale, self.network.scale], dtype=int)
        seq = list(range(self.network.scale))
        random.shuffle(seq)
        for i in range(self.network.scale):
            T[i][seq[i]] = 1

        self.grouping_matrix = np.empty([self.P, self.network.scale, self.network.scale])
        for i in range(self.P):
            self.grouping_matrix[i] = np.dot(np.dot(T, self.block_diagonal_matrix), T)

    def cal_payoff(self):
        """Calculate total payoff.
        """
        # Calculate motivation after grouping.
        for idx in range(self.network.scale):
            self.network.agents[idx].motivation += \
                self.grouping_matrix[idx,:] * self.network.T1[:, idx] 
        # Calculate skill promotion.

    def write2file(self, path):
        """将分组结果写入磁盘文件

        Args:
            path (_type_): _description_
        """
        pass

    # def random(self):
    #     """Randomly assign agents.
    #     """
    #     # Shuffle the set of agents
    #     agents = self.network.agents.copy()
    #     random.shuffle(agents)

    #     group_num = math.ceil(self.network.scale / Group.max_member_num) # the number of groups
    #     for idx in range(group_num):
    #         self.groups.append(Group(idx))
    #     for i in range(self.network.scale):
    #         self.groups[i % group_num].add_agent(agents[i])
        