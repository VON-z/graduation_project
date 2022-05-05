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
from multiplex_network.network import Network
from ga import GA

# Hyperparameters
SKILL_NUM = 2
MAX_MEMBER_NUM = 2
NETWORK_SCALE = 4
POPULATION = 3
PC = 0.5
PM = 0.01
IT = 10

# Initialize Agent class variables.
Agent.skill_num = SKILL_NUM

# Build the network.
network = Network(NETWORK_SCALE)

# Genetic Algorithm.
algorithm = GA(network, MAX_MEMBER_NUM, SKILL_NUM, POPULATION, PC, PM, IT)

# Generate initial population.
algorithm.generate_initial_population()
# Calculate evaluation.
for p in range(POPULATION):
    algorithm.cal_payoff()
    algorithm.cal_evaluation()
