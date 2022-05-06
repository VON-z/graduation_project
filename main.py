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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

# here put the local import source
from multiplex_network.agent import Agent
from multiplex_network.network import Network
from ga import GA

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5
NETWORK_SCALE = 100
POPULATION = 100
PC = 0.5
PM = 0.1
IT = 1000

# Initialize Agent class variables.
Agent.skill_num = SKILL_NUM

# Build the network.
network = Network(NETWORK_SCALE)

# Genetic Algorithm.
algorithm = GA(network, MAX_MEMBER_NUM, SKILL_NUM, POPULATION, PC, PM, IT)

# Generate initial population.
algorithm.generate_initial_population()

evaluation = np.empty([IT, POPULATION])
for generation_num in trange(IT):
    # Calculate payoff tensor.
    algorithm.cal_payoff()
    # Calculate evaluation.
    algorithm.cal_evaluation()
    evaluation[generation_num] = algorithm.evaluation.copy()    # draw heatmap.
    # Calculate fitness.
    algorithm.cal_fitness()

    # Selection.
    algorithm.selection()
    # Recombination.
    algorithm.recombination()
    # Mutation.
    algorithm.mutation()

# Draw evaluation figure.
# 支持中文以及负数
plt.rcParams['font.sans-serif']=['STSong']  # SimHei黑体 STSong宋体
plt.rcParams['axes.unicode_minus'] = False

x = list(range(IT))
y = [sum(evaluation[i]) / POPULATION for i in range(IT)]
plt.subplot(1, 2, 1)
plt.plot(x, y, linewidth=2.0)
# plt.show()

# Draw heatmap。
plt.subplot(1, 2, 2)
sns.set_theme()
sns.heatmap(evaluation.T)
plt.show()
print('test')