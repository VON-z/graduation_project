# -*- encoding: utf-8 -*-
'''
@File    :   Network.py
@Time    :   2022/03/23 13:48:00
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   None
'''

# here put the standard library

# here put the third-party packages
import numpy as np

# here put the local import source
from .agent import Agent

class Network():
    """Multiplex network modeling.
    Including the maximum scale of the network, the current scale of the network, 
    the set of agents, and two adjacency matrices (w1, w2).
    """

    def __init__(self, scale, **kw) -> None:
        """Initialize the multiplex network.

        Args:
            optional:
                agents (list): the set of agents.
                w1 (ndarray): the motivation network adjacenecy matrix.
                w2 (ndarray): the skill network adjacency matrix.
        """
        self.scale = scale
        self.agents = []
        self.W1 = None
        self.W2 = None
        self.T1 = None  # Temporary matrix used to calculate the motivation promotion.
        self.T2 = None  # Temporary tensor used to calculate the skill promotion.

        # Network initialization.
        if 'agents' in kw:
            # Data initialization.
            self.agents = kw['agents'].copy()
        else:
            # Random initialization.
            self.generate_agent_set()

        if 'W1' in kw:
            # Data initialization.
            self.W1 = kw['W1'].copy()
        else:
            # Random initialization.
            self.generate_w1()

        if 'W2' in kw:
            # Data initialization
            self.W2 = kw['W2'].copy()
        else:
            # Random initialization.
            self.generate_w2()

        # Calculate temporary matrix.
        self.cal_t1()
        self.cal_t2()

    def generate_agent_set(self):
        """Randomly generate a set of agents.
        """
        for idx in range(self.scale):
            a = Agent(idx)
            self.agents.append(a)

    def generate_w1(self):
        """Randomly generate W1 matrix.
        """
        self.W1 = np.random.rand(self.scale, self.scale)

    def generate_w2(self):
        """Randomly generate W2 matrix.
        """
        self.W2 = np.random.rand(self.scale, self.scale)

    def cal_t1(self):
        """Calculate temporary matrix used to calculate the motivation promotion.
        """
        self.T1 = np.empty([self.scale, self.scale])
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                self.T1[i][j] = (a1.motivation - a2.motivation) * self.W1[i][j]

    def cal_t2(self):
        """Calculate temporary tensor used to calculate the skill promotion.
        """
        self.T2 = np.empty([Agent.skill_num,self.scale, self.scale])
        for k in range(Agent.skill_num):
            for i, a1 in enumerate(self.agents):
                for j, a2 in enumerate(self.agents):
                    self.T2[k][i][j] = (a1.skills[k] - a2.skills[k]) * self.W2[i][j]

                    # In terms of skills, we only considered the improvement
                    # of the weak from the strong.
                    if self.T2[k][i][j] < 0:
                        self.T2[k][i][j] = 0
                        
    def write2file(self, path):
        """将网络参数写入磁盘文件

        Args:
            path (_type_): _description_
        """
        pass
