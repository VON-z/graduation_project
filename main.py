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
from pickle import POP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

# here put the local import source
from multiplex_network.agent import Agent
from multiplex_network.network import Network
from ga import GA
from sa import SA
from data import load_skills
from data import load_motivation
from data import load_network
from data import write2file

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5

# Network Hyperparameters
NETWORK_NUM = 10
NETWORK_SCALE = [20, 60, 100]
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

Agent.skill_num = SKILL_NUM # Initialize Agent class variables.  

# 1.Genetic Algorithm.
# for r in trange(10):
#     for network_scale in NETWORK_SCALE:
#         skills_matrix = load_skills(network_scale, r)
#         motivation = load_motivation(network_scale, r)
#         # ER
#         for p in ER_P:
#             w1 = load_network(network_scale, 'ER', r=r, layer=1, p=p)
#             w2 = load_network(network_scale, 'ER', r=r, layer=2, p=p)
#             network = Network(network_scale, W1=w1, W2=w2,
#                 skills=skills_matrix, motivation=motivation)    # Build the network.
#             ga = GA(network, MAX_MEMBER_NUM, SKILL_NUM)
#             # Execute.
#             for pc in PC:
#                 for pm in PM:
#                     evaluation, best_solution, best_solution_evaluation = \
#                         ga(C_max=C_MAX, P=POPULATION, pc=pc, pm=pm)
#                     write2file(evaluation, best_solution, best_solution_evaluation,
#                         r, 'GA', 'ER', network_scale, p=p, pc=pc, pm=pm)

#         # WS
#         for p in WS_P:
#             w1 = load_network(network_scale, 'WS', r=r, layer=1, p=p)
#             w2 = load_network(network_scale, 'WS', r=r, layer=2, p=p)
#             network = Network(network_scale, W1=w1, W2=w2,
#                 skills=skills_matrix, motivation=motivation)    # Build the network.
#             ga = GA(network, MAX_MEMBER_NUM, SKILL_NUM)
#             # Execute.
#             for pc in PC:
#                 for pm in PM:
#                     evaluation, best_solution, best_solution_evaluation = \
#                         ga(C_max=C_MAX, P=POPULATION, pc=pc, pm=pm)
#                     write2file(evaluation, best_solution, best_solution_evaluation,
#                         r, 'GA', 'WS', network_scale, p=p, pc=pc, pm=pm)

#         # BA
#         for m in BA_M:
#             w1 = load_network(network_scale, 'BA', r=r, layer=1, m=m)
#             w2 = load_network(network_scale, 'BA', r=r, layer=2, m=m)
#             network = Network(network_scale, W1=w1, W2=w2,
#                 skills=skills_matrix, motivation=motivation)    # Build the network.
#             ga = GA(network, MAX_MEMBER_NUM, SKILL_NUM)
#             # Execute.
#             for pc in PC:
#                 for pm in PM:
#                     evaluation, best_solution, best_solution_evaluation = \
#                         ga(C_max=C_MAX, P=POPULATION, pc=pc, pm=pm)
#                     write2file(evaluation, best_solution, best_solution_evaluation,
#                         r, 'GA', 'BA', network_scale, m=m, pc=pc, pm=pm)


# 2.Simulated Annealing.
for r in trange(50):
    for network_scale in NETWORK_SCALE:
        skills_matrix = load_skills(network_scale, r)
        motivation = load_motivation(network_scale, r)
        # ER
        for p in ER_P:
            w1 = load_network(network_scale, 'ER', r=r, layer=1, p=p)
            w2 = load_network(network_scale, 'ER', r=r, layer=2, p=p)
            network = Network(network_scale, W1=w1, W2=w2,
                skills=skills_matrix, motivation=motivation)    # Build the network.
            sa = SA(network, MAX_MEMBER_NUM, SKILL_NUM)
            # Execute.
            for t in TE:
                for alpha in ALPHA:
                    evaluation, best_solution, best_solution_evaluation = \
                        sa(C_max=C_MAX, T=t, alpha=alpha, L=L)
                    write2file(evaluation, best_solution, best_solution_evaluation,
                        r, 'SA', 'ER', network_scale, p=p, t=t, alpha=alpha)

        # WS
        for p in WS_P:
            w1 = load_network(network_scale, 'WS', r=r, layer=1, p=p)
            w2 = load_network(network_scale, 'WS', r=r, layer=2, p=p)
            network = Network(network_scale, W1=w1, W2=w2,
                skills=skills_matrix, motivation=motivation)    # Build the network.
            sa = SA(network, MAX_MEMBER_NUM, SKILL_NUM)
            # Execute.
            for t in TE:
                for alpha in ALPHA:
                    evaluation, best_solution, best_solution_evaluation = \
                        sa(C_max=C_MAX, T=t, alpha=alpha, L=L)
                    write2file(evaluation, best_solution, best_solution_evaluation,
                        r, 'SA', 'WS', network_scale, p=p, t=t, alpha=alpha)

        # BA
        for m in BA_M:
            w1 = load_network(network_scale, 'BA', r=r, layer=1, m=m)
            w2 = load_network(network_scale, 'BA', r=r, layer=2, m=m)
            network = Network(network_scale, W1=w1, W2=w2,
                skills=skills_matrix, motivation=motivation)    # Build the network.
            sa = SA(network, MAX_MEMBER_NUM, SKILL_NUM)
            # Execute.
            for t in TE:
                for alpha in ALPHA:
                    evaluation, best_solution, best_solution_evaluation = \
                        sa(C_max=C_MAX, T=t, alpha=alpha, L=L)
                    write2file(evaluation, best_solution, best_solution_evaluation,
                        r, 'SA', 'BA', network_scale, m=m, t=t, alpha=alpha)
