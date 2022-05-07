# -*- encoding: utf-8 -*-
'''
@File    :   generate_data.py
@Time    :   2022/05/06 15:57:53
@Author  :   VONz 
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   Generate and read data for experiments, 
             including agent skills, motivation and network weights.
'''

# here put the standard library
import random
import math
import os

# here put the third-part
import networkx as nx
import numpy as np

# here put the local import source
from multiplex_network.agent import Agent
from multiplex_network.network import Network
from ga import GA

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5
NETWORK_SCALE = [100, 200, 500]

# 1.Generate
# Agent skills and motivation.
def generate_skills(network_scale, path):
    """Generate skills.

    Args:
        network_scale (int): number of agents.
        path (string): storage path.

    Returns:
        ndarray: skills matrix.
    """
    skills = np.random.beta(4, 4, (network_scale, SKILL_NUM))
    if path:
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(os.path.join(path, 'skills_n{}'.format(network_scale)), skills)
    return skills

def generate_motivation(network_scale, path):
    """Generate motivation.

    Args:
        network_scale (int): number of agents.
        path (string): storage path.

    Returns:
        ndarray: motivation matrix.
    """
    motivation = np.random.beta(4, 4, network_scale)
    if path:
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(os.path.join(path, 'motivation_n{}'.format(network_scale)), motivation)
    return motivation

# Network weights.
def generate_er(network_scale, p, i, path):
    """Generate ER graph.

    Args:
        network_scale (int): number of agents.
        p (float): edge probability.
        i (int): network layer serial number.
        path (string): storage path.

    Returns:
        ndarray: weights matrix.
    """
    weights = np.zeros([network_scale, network_scale])
    # Generate ER graph.
    er_graph = nx.random_graphs.erdos_renyi_graph(network_scale, p, directed=True)
    # Generate weights.
    for edge in er_graph.edges:
        weights[edge[0]][edge[1]] = random.random()
    # Write to disk.
    if path:
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(os.path.join(path, 'er{}_n{}_p{}'.format(i, network_scale, p)), weights)
    return weights

def generate_ws(network_scale, k, p, i, path):
    """Generate WS small world graph.

    Args:
        network_scale (int): number of agents.
        k (int): each node is joined with its k nearest neighbors in a ring topology.
        p (float): the probability of rewiring each edge.
        i (int): network layer serial number.
        path (string): storage path.

    Returns:
        ndarray: weights matrix.
    """
    weights = np.zeros([network_scale, network_scale])
    # Generate WS graph.
    ws_graph = nx.random_graphs.watts_strogatz_graph(network_scale, k, p)
    # Generate weights.
    for edge in ws_graph.edges:
        weights[edge[0]][edge[1]] = random.random()
    # Write to disk.
    if path:
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(os.path.join(path, 'ws{}_n{}_k{}_p{}'.format(i, network_scale, MAX_MEMBER_NUM, p)), weights)
    return weights

def generate_ba(network_scale, m, i, path):
    """Generate BA scale free graph.
    Args:
        network_scale (int): number of agents.
        m (int): number of edges to attach from a new node to existing nodes.
        i (int): network layer serial number.
        path (string): storage path.

    Returns:
        ndarray: weights matrix.
    """
    weights = np.zeros([network_scale, network_scale])
    # Generate WS graph.
    ba_graph = nx.random_graphs.barabasi_albert_graph(network_scale, m)
    # Generate weights.
    for edge in ba_graph.edges:
        weights[edge[0]][edge[1]] = random.random()
    # Write to disk.
    if path:
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(os.path.join(path, 'ba{}_n{}_m{}'.format(i, network_scale, m)), weights)
    return weights

# 2.Load
def load_skills(network_scale):
    """Load skills.

    Args:
        network_scale (int): number of agents.

    Returns:
        ndarray: skills matrix.
    """
    path = os.path.join('./data', 'skills', 'skills_n{}.npy'.format(network_scale))
    skills = np.load(path)
    return skills

def load_motivation(network_scale):
    """Load motivation.

    Args:
        network_scale (int): number of agents.

    Returns:
        ndarray: motivation matrix.
    """
    path = os.path.join('./data', 'motivation', 'motivation_n{}.npy'.format(network_scale))
    motivation = np.load(path)
    return motivation

def load_network(network_scale, network_type, **kw):
    """Load different type network.

    Args:
        network_scale (int): number of agents.
        network_type (string): network type.
        optional:
            layer (int): network layer serial number.
            p (float): probability
            m (int): number of edges to attach from a new node to existing nodes.
    """
    if network_type == 'ER':
        path = os.path.join('./data', 'network', 'ER', 
            'er{}_n{}_p{}'.format(kw['layer'], network_scale, kw['p']))
        weights = np.load(path)
        return weights

    if network_type == 'WS':
        path = os.path.join('./data', 'network', 'WS', 
            'ws{}_n{}_k{}_p{}'.format(kw['layer'], network_scale, MAX_MEMBER_NUM, kw['p']))
        weights = np.load(path)
        return weights

    if network_type == 'BA':
        path = os.path.join('./data', 'network', 'BA', 
            'ba{}_n{}_m{}'.format(kw['layer'], network_scale, kw['m']))
        weights = np.load(path)
        return weights


# if __name__=='__main__':
#     for n in NETWORK_SCALE:
#         # Agent skills and motivation.
#         generate_skills(n, os.path.join('data', 'skills'))
#         generate_motivation(n, os.path.join('data', 'motivation'))

#         # Network weights.
#         # ER
#         for p in [1/n, math.ceil(math.log(n))/n]:
#             for i in [1, 2]:
#                 generate_er(n, p, i, os.path.join('./data', 'network', 'ER'))
#         # WS
#         for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
#             for i in [1, 2]:
#                 generate_ws(n, MAX_MEMBER_NUM, p, i, os.path.join('./data', 'network', 'WS'))
#         # BA
#         for m in [1,3,5,7]:
#             for i in [1, 2]:
#                 generate_ba(n, m, i, os.path.join('./data', 'network', 'BA'))

# Test load function.
for n in NETWORK_SCALE:
    skills = load_skills(n)
    motivation = load_motivation(n)

    print('breakpoint')
        