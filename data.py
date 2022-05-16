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
from tqdm import trange

# here put the local import source

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5

# Network Hyperparameters
NETWORK_SCALE = [20, 40, 60, 80, 100]
NETWORK_TYPE = ['ER', 'WS', 'BA']
ER_P = [0.1, 0.5, 0.9]
WS_P = [0.1, 0.5, 0.9]
BA_M = [1, 5, 9]

# Genetic Algorithm Parameters.
C_MAX = 500
POPULATION = 100
PC = [0.3, 0.5, 0.7]
PM = [0.05, 0.1, 0.2]

# Simulated Annealing Parameters.
TE = [10, 100, 500]
ALPHA = [0.95, 0.97, 0.99]
L = 5

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
        if not os.path.exists(path):
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
        if not os.path.exists(path):
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
        if not os.path.exists(path):
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
        if not os.path.exists(path):
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
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'ba{}_n{}_m{}'.format(i, network_scale, m)), weights)
    return weights

# 2.Load
def load_skills(network_scale, r):
    """Load skills.

    Args:
        network_scale (int): number of agents.
        r (int): the r-th round experiment of the same parameter network.

    Returns:
        ndarray: skills matrix.
    """
    path = os.path.join('./data', str(r), 'skills', 'skills_n{}.npy'.format(network_scale))
    skills = np.load(path)
    return skills

def load_motivation(network_scale, r):
    """Load motivation.

    Args:
        network_scale (int): number of agents.
        r (int): the r-th round experiment of the same parameter network.

    Returns:
        ndarray: motivation matrix.
    """
    path = os.path.join('./data', str(r), 'motivation', 'motivation_n{}.npy'.format(network_scale))
    motivation = np.load(path)
    return motivation

def load_network(network_scale, network_type, **kw):
    """Load different type network.

    Args:
        network_scale (int): number of agents.
        network_type (string): network type.
        optional:
            r (int): the r-th round experiment of the same parameter network.
            layer (int): network layer serial number.
            p (float): probability
            m (int): number of edges to attach from a new node to existing nodes.
    """
    if network_type == 'ER':
        path = os.path.join('./data', str(kw['r']), 'network', 'ER', 
            'er{}_n{}_p{}.npy'.format(kw['layer'], network_scale, kw['p']))
        weights = np.load(path)
        return weights

    if network_type == 'WS':
        path = os.path.join('./data', str(kw['r']), 'network', 'WS', 
            'ws{}_n{}_k{}_p{}.npy'.format(kw['layer'], network_scale, MAX_MEMBER_NUM, kw['p']))
        weights = np.load(path)
        return weights

    if network_type == 'BA':
        path = os.path.join('./data', str(kw['r']), 'network', 'BA', 
            'ba{}_n{}_m{}.npy'.format(kw['layer'], network_scale, kw['m']))
        weights = np.load(path)
        return weights

# 3.Save experiment data.
def write2file(evaluation, best_solution, best_solution_evaluation,
    r, alg, network_type, network_scale, **kw):
    """_summary_

    Args:
        evaluation (list or ndarray): _description_
        best_solution (ndarray): _description_
        best_solution_evaluation (float): _description_
        r (int): _description_
        alg (str): algorithm type.
        network_scale (int): _description_
        network_type (str): _description_
    """
    if alg == 'GA':
        if network_type == 'ER':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'er_n{}_p{}_pc{}_pm{}_eva'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), evaluation)
            np.save(os.path.join(path, 'er_n{}_p{}_pc{}_pm{}_bs'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), best_solution)
            np.save(os.path.join(path, 'er_n{}_p{}_pc{}_pm{}_bse'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), best_solution_evaluation)

        if network_type == 'WS':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'ws_n{}_p{}_pc{}_pm{}_eva'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), evaluation)
            np.save(os.path.join(path, 'ws_n{}_p{}_pc{}_pm{}_bs'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), best_solution)
            np.save(os.path.join(path, 'ws_n{}_p{}_pc{}_pm{}_bse'.format(
                network_scale, kw['p'], kw['pc'], kw['pm'])), best_solution_evaluation)

        if network_type == 'BA':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'ba_n{}_m{}_pc{}_pm{}_eva'.format(
                network_scale, kw['m'], kw['pc'], kw['pm'])), evaluation)
            np.save(os.path.join(path, 'ba_n{}_m{}_pc{}_pm{}_bs'.format(
                network_scale, kw['m'], kw['pc'], kw['pm'])), best_solution)
            np.save(os.path.join(path, 'ba_n{}_m{}_pc{}_pm{}_bse'.format(
                network_scale, kw['m'], kw['pc'], kw['pm'])), best_solution_evaluation)

    elif alg == 'SA':
        if network_type == 'ER':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'er_n{}_p{}_t{}_alpha{}_eva'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), evaluation)
            np.save(os.path.join(path, 'er_n{}_p{}_t{}_alpha{}_bs'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), best_solution)
            np.save(os.path.join(path, 'er_n{}_p{}_t{}_alpha{}_bse'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), best_solution_evaluation)

        if network_type == 'WS':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'ws_n{}_p{}_t{}_alpha{}_eva'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), evaluation)
            np.save(os.path.join(path, 'ws_n{}_p{}_t{}_alpha{}_bs'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), best_solution)
            np.save(os.path.join(path, 'ws_n{}_p{}_t{}_alpha{}_bse'.format(
                network_scale, kw['p'], kw['t'], kw['alpha'])), best_solution_evaluation)

        if network_type == 'BA':
            path = os.path.join('./result', str(r), alg, network_type)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'ba_n{}_m{}_t{}_alpha{}_eva'.format(
                network_scale, kw['m'], kw['t'], kw['alpha'])), evaluation)
            np.save(os.path.join(path, 'ba_n{}_m{}_t{}_alpha{}_bs'.format(
                network_scale, kw['m'], kw['t'], kw['alpha'])), best_solution)
            np.save(os.path.join(path, 'ba_n{}_m{}_t{}_alpha{}_bse'.format(
                network_scale, kw['m'], kw['t'], kw['alpha'])), best_solution_evaluation)


# 4.Load experiment data.
def load_experiment_data(r, alg, network_type, network_scale, **kw):
    """_summary_

    Args:
        r (int): _description_
        alg (str): _description_
        network_type (str): _description_

    Returns:
        _type_: _description_
    """
    evaluation = None
    best_solution = None
    best_solution_evaluation = None

    path = os.path.join('./result', str(r), alg, network_type)
    pref = None
    if alg == 'GA':
        if network_type == 'ER':
            pref = 'er_n{}_p{}_pc{}_pm{}_'.format(network_scale, kw['p'], kw['pc'], kw['pm'])
        elif network_type == 'WS':
            pref = 'ws_n{}_p{}_pc{}_pm{}_'.format(network_scale, kw['p'], kw['pc'], kw['pm'])
        elif network_type == 'BA':
            pref = 'ba_n{}_m{}_pc{}_pm{}_'.format(network_scale, kw['m'], kw['pc'], kw['pm'])

    elif alg == 'SA':
        if network_type == 'ER':
            pref = 'er_n{}_p{}_t{}_alpha{}_'.format(network_scale, kw['p'], kw['t'], kw['alpha'])
        elif network_type == 'WS':
            pref = 'ws_n{}_p{}_t{}_alpha{}_'.format(network_scale, kw['p'], kw['t'], kw['alpha'])
        elif network_type == 'BA':
            pref = 'ba_n{}_m{}_t{}_alpha{}_'.format(network_scale, kw['m'], kw['t'], kw['alpha'])

    evaluation = np.load(os.path.join(path, ''.join([pref, 'eva.npy'])))
    best_solution = np.load(os.path.join(path, ''.join([pref, 'bs.npy'])))
    best_solution_evaluation = np.load(os.path.join(path, ''.join([pref, 'bse.npy'])))
    return evaluation.copy(), best_solution.copy(), best_solution_evaluation.copy()

# 5.Organize data into tables.
def generate_data_table(network_type, **kw):
    """_summary_

    Args:
        network_type (str): _description_
    """
    data_table = np.empty([10, 9])
    # ER
    if network_type == 'ER':
        ## SA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, t in enumerate(TE):
                for k, alpha in enumerate(ALPHA):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'SA', network_type, network_scale,
                        p=kw['p'], t=t, alpha=alpha)
                        bse_all.append(bse)
                    data_table[i][j*3+k] = sum(bse_all) / 10
        ## GA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, pc in enumerate(PC):
                for k, pm in enumerate(PM):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'GA', network_type, network_scale,
                        p=kw['p'], pc=pc, pm=pm)
                        bse_all.append(bse)
                    data_table[i+len(NETWORK_SCALE)][j*3+k] = sum(bse_all) / 10
        path = './table'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'er_p{}.csv'.format(kw['p'])), data_table, delimiter=',')

    # WS
    elif network_type == 'WS':
        ## SA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, t in enumerate(TE):
                for k, alpha in enumerate(ALPHA):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'SA', network_type, network_scale,
                        p=kw['p'], t=t, alpha=alpha)
                        bse_all.append(bse)
                    data_table[i][j*3+k] = sum(bse_all) / 10
        ## GA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, pc in enumerate(PC):
                for k, pm in enumerate(PM):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'GA', network_type, network_scale,
                        p=kw['p'], pc=pc, pm=pm)
                        bse_all.append(bse)
                    data_table[i+len(NETWORK_SCALE)][j*3+k] = sum(bse_all) / 10
        path = './table'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'ws_p{}.csv'.format(kw['p'])), data_table, delimiter=',')
    # BA
    elif network_type == 'BA':
        ## SA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, t in enumerate(TE):
                for k, alpha in enumerate(ALPHA):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'SA', network_type, network_scale,
                        m=kw['m'], t=t, alpha=alpha)
                        bse_all.append(bse)
                    data_table[i][j*3+k] = sum(bse_all) / 10
        ## GA
        for i, network_scale in enumerate(NETWORK_SCALE):
            for j, pc in enumerate(PC):
                for k, pm in enumerate(PM):
                    bse_all = []
                    for r in range(10):
                        _, _, bse = load_experiment_data(r, 'GA', network_type, network_scale,
                        m=kw['m'], pc=pc, pm=pm)
                        bse_all.append(bse)
                    data_table[i+len(NETWORK_SCALE)][j*3+k] = sum(bse_all) / 10
        path = './table'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'ba_m{}.csv'.format(kw['m'])), data_table, delimiter=',')

# Generate data.
# if __name__=='__main__':
#     for r in trange(NETWORK_NUM):
#         for n in NETWORK_SCALE:
#             # Agent skills and motivation.
#             generate_skills(n, os.path.join('./data', str(r), 'skills'))
#             generate_motivation(n, os.path.join('./data', str(r), 'motivation'))

#             # Network weights.
#             # ER
#             for p in ER_P:
#                 for i in [1, 2]:
#                     generate_er(n, p, i, os.path.join('./data', str(r), 'network', 'ER'))
#             # WS
#             for p in WS_P:
#                 for i in [1, 2]:
#                     generate_ws(n, MAX_MEMBER_NUM, p, i, 
#                         os.path.join('./data', str(r), 'network', 'WS'))
#             # BA
#             for m in BA_M:
#                 for i in [1, 2]:
#                     generate_ba(n, m, i, os.path.join('./data', str(r), 'network', 'BA'))

# Organize data into tables.
if __name__ == "__main__":
    ## ER
    for p in ER_P:
        generate_data_table('ER', p=p)
    ## WS
    for p in WS_P:
        generate_data_table('WS', p=p)
    ## BA
    for m in BA_M:
        generate_data_table('BA', m=m)
