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
import math
import random

# here put the third-party packages
import scipy.linalg
import numpy as np

# here put the local import source

# from multiplex_network.network import Network


class GA():
    """Implement different algorithms.
    Including grouping results and total payoff.
    """
    def __init__(self, network, group_scale, skill_num, P, pc, pm, it) -> None:
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
        self.skill_num = skill_num
        self.P = P
        self.pc = pc
        self.pm = pm
        self.it = it

        self.block_diagonal_matrix = None
        self.generate_block_diagonal_matrix()

        self.grouping_tensor = None # The tensor representing a series of grouping results。
        self.payoff_tensor = None   # The tensor records payoff of a series of grouping results.
        self.evaluation = []
        self.fitness = []
        self.intermediate_generation = []

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
            blocks.append(np.ones([self.group_scale-1, self.group_scale-1], dtype=int))
        self.block_diagonal_matrix = scipy.linalg.block_diag(*blocks)

    def generate_initial_population(self):
        """Generate initial population.
        """
        self.grouping_tensor = np.empty([self.P, self.network.scale, self.network.scale])
        for p in range(self.P):
            T = np.zeros([self.network.scale, self.network.scale], dtype=int)
            seq = list(range(self.network.scale))
            random.shuffle(seq)
            for i in range(self.network.scale):
                T[i][seq[i]] = 1

            self.grouping_tensor[p] = np.dot(np.dot(T, self.block_diagonal_matrix), T)

    def cal_payoff(self):
        """Calculate payoff for each agent in each skill of all grouping results.
        """
        self.payoff_tensor = np.empty([self.P, self.network.scale, self.skill_num])
        for p in range(self.P):
            # Calculate motivation after grouping.
            for idx in range(self.network.scale):
                self.network.agents[idx].motivation += np.dot(
                    self.grouping_tensor[p,idx,:], self.network.T1[:, idx]
                )

            # Calculate skill promotion.
            for idx in range(self.network.scale):
                for k in range(self.skill_num):
                    self.payoff_tensor[p][idx][k] = np.dot(self.grouping_tensor[p,idx,:],
                    self.network.T2[k,:,idx]) * self.network.agents[idx].motivation

    def cal_evaluation(self):
        """Calculate the evaluation of each population.
        """
        for p in range(self.P):
            self.evaluation.append(np.sum(self.payoff_tensor[p,:,:]))

    def cal_fitness(self):
        """Calculate the fitness of each solution.
        """
        avg_fitness = sum(self.evaluation) / len(self.evaluation)
        for item in self.evaluation:
            self.fitness.append(item / avg_fitness)

    def selection(self):
        """Selection is applied to the current population
        to create an intermediate population.
        """
        for i, fitness in enumerate(self.fitness):
            # Integer portion
            direct_placed = math.floor(fitness)
            for _ in range(direct_placed):
                self.intermediate_generation.append(self.grouping_tensor[i])
            # Fractional portion
            probability_placed = fitness - direct_placed
            if random.random() < probability_placed:
                self.intermediate_generation.append(self.grouping_tensor[i])

        # shuffle
        random.shuffle(self.intermediate_generation)

    def crossover(self):
        """Implement crossover function.
        """
        

    def mutation(self, idx):
        """Implement mutation function.

        Args:
            idx (int): the index of the solution to be mutated.
        """
        pass

    def write2file(self, path):
        """Write to disk file.

        Args:
            path (_type_): _description_
        """
        pass

        