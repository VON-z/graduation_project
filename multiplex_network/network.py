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
    Including the number of network layers, the maximum scale of the network,
    the current scale of the network, the set of agents, the adjacency matrix
    and motivation network.
    """
    layers_num = 0  # the number of network layers.
    max_scale = 0   # the maximum scale of the network.

    def __init__(self, **kw) -> None:
        """Initialize the multiplex network.

        Args:
            optional:
                agents (list): the set of agents.
                adjacency_matrix (ndarray): the adjacency matrix.
                motivation_network (ndarray): the motivation network adjacenecy matrix.
        """
        self.scale = 0
        self.agents = []
        self.adjacency_matrix = None
        self.motivation_network = None

        if 'agents' in kw:
            # Data initialization.
            self.agents = kw['agents'].copy()
        else:
            # Random initialization.
            self.generate_agent_set()

        if 'adjacency_matrix' in kw:
            # Data initialization.
            self.adjacency_matrix = kw['adjacency_matrix'].copy()
        else:
            # Random initialization.
            self.generate_network_weights()

        if 'motivation_network' in kw:
            # Data initialization
            self.motivation_network = kw['motivation_network'].copy()
        else:
            # Random initialization.
            self.generate_motivation_network()

    def add_agent(self, a):
        """Add an agent.

        Args:
            a (Agent): Agent instance to be added.
        """
        if self.scale < Network.max_scale:
            self.agents.append(a)
            self.scale += 1

    def generate_agent_set(self):
        """Randomly generate a set of agents.
        """
        for idx in range(self.max_scale):
            a = Agent(idx)
            self.add_agent(a)

    def generate_network_weights(self):
        """Randomly generate the edge weights of a network.
        (过渡函数,后续要细化随机生成哪种复杂网络,BA,WS...)
        """
        self.adjacency_matrix = np.random.rand(
            Network.layers_num, Network.max_scale, Network.max_scale)

    def generate_motivation_network(self):
        """Randomly generate motivation network.
        (过渡函数,后续要细化随机生成哪种复杂网络,BA,WS...)
        """
        self.motivation_network = np.random.rand(
            1, Network.max_scale, Network.max_scale)

    def write2file(self, path):
        """将网络参数写入磁盘文件

        Args:
            path (_type_): _description_
        """
        pass
