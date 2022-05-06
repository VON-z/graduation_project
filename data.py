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
NETWORK_SCALE = [100, 300, 500]

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
    np.save(path, skills)
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
    np.save(path, motivation)
    return motivation

# Network weights.
def generate_er(network_scale, p, path):
    """Generate ER graph.

    Args:
        network_scale (int): number of agents.
        p (float): edge probability.
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
        np.save(path, weights)
    return weights

def generate_ws(network_scale, k, p, path):
    """Generate WS small world graph.

    Args:
        network_scale (int): number of agents.
        k (int): each node is joined with its k nearest neighbors in a ring topology.
        p (float): the probability of rewiring each edge.
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
        np.save(path, weights)
    return weights

def generate_ba(network_scale, m, path):
    """Generate BA scale free graph.
    Args:
        network_scale (int): number of agents.
        m (int): number of edges to attach from a new node to existing nodes.
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
        np.save(path, weights)
    return weights

# 2.Load

if __name__=='__main__':
    for n in NETWORK_SCALE:
        # Agent skills and motivation.
        generate_skills(n, os.path.join('data', 'skills', 'skills_n{}'.format(n)))
        generate_motivation(n, os.path.join('data', 'motivation', 'motivation_n{}'.format(n)))

        # Network weights.
        # ER
        for p in [1/n, math.log(n)/n]:
            generate_er(n, p, os.path.join('data', 'network', 'er1_n{}_p{}'.format(n, p)))
            generate_er(n, p, os.path.join('data', 'network', 'er2_n{}_p{}'.format(n, p)))
        # WS
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            generate_ws(n, MAX_MEMBER_NUM, p, os.path.join(
                'data', 'network', 'ws1_n{}_k{}_p{}'.format(n, MAX_MEMBER_NUM, p)))
            generate_ws(n, MAX_MEMBER_NUM, p, os.path.join(
                'data', 'network', 'ws2_n{}_k{}_p{}'.format(n, MAX_MEMBER_NUM, p)))
        # BA
        for m in [1,3,5,7]:
            generate_ba(n, m,  os.path.join('data', 'network', 'ba1_n{}_m{}'.format(n, m)))
            generate_ba(n, m,  os.path.join('data', 'network', 'ba2_n{}_m{}'.format(n, m)))

        