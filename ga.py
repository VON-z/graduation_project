# -*- encoding: utf-8 -*-
'''
@File    :   algorithm.py
@Time    :   2022/03/27 13:33:46
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   Implement genetic algorithm to solve the problem.
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
    """Implement genetic algorithm.
    """
    def __init__(self, network, group_scale, skill_num, P, pc, pm, it) -> None:
        """Initialization.

        Args:
            network (Network): network model.
            group_scale (int): max member number in a group.
            skill_num (int): skill number.
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
        group_num = math.ceil(self.network.scale / self.group_scale)
        missing = group_num * self.group_scale - self.network.scale
        complete = group_num - missing

        # Generate block diagonal matrix.
        blocks = []
        for _ in range(complete):
            blocks.append(np.ones([self.group_scale, self.group_scale], dtype=int))
        for _ in range(missing):
            blocks.append(np.ones([self.group_scale-1, self.group_scale-1], dtype=int))
        self.block_diagonal_matrix = scipy.linalg.block_diag(*blocks)
        self.block_diagonal_matrix[np.eye(self.network.scale, dtype=bool)] = 0

    def generate_initial_population(self):
        """Generate initial population.
        """
        self.grouping_tensor = np.empty([self.P, self.network.scale, self.network.scale])
        for p in range(self.P):
            self.grouping_tensor[p] = self.block_diagonal_matrix.copy()
            for i in range(self.network.scale):
                j = random.randint(0, self.network.scale-1)
                if sum(self.grouping_tensor[p][i] * self.grouping_tensor[p][j]) < 0.5:
                    # Connecting agents are completely different.
                    # Row swap.
                    self.grouping_tensor[p][i], self.grouping_tensor[p][j] = \
                        self.grouping_tensor[p][j].copy(), self.grouping_tensor[p][i].copy()
                    # Column swap.
                    self.grouping_tensor[p,:,i], self.grouping_tensor[p,:,j] = \
                        self.grouping_tensor[p,:,j].copy(), self.grouping_tensor[p,:,i].copy()

    def cal_payoff(self):
        """Calculate payoff for each agent in each skill of all grouping results.
        """
        self.payoff_tensor = np.empty([self.P, self.network.scale, self.skill_num])
        for p in range(self.P):
            new_motivation = []
            # Calculate motivation after grouping.
            for idx in range(self.network.scale):
                neighbor_num = np.count_nonzero(
                    self.grouping_tensor[p,idx,:] * self.network.T1[:, idx]
                )
                if neighbor_num:
                    new_motivation.append(
                        self.network.agents[idx].motivation + np.dot(
                            self.grouping_tensor[p,idx,:], self.network.T1[:, idx]) / neighbor_num
                    )
                else:
                    new_motivation.append(self.network.agents[idx].motivation)

            # Calculate skill promotion.
            for idx in range(self.network.scale):
                for k in range(self.skill_num):
                    neighbor_num = np.count_nonzero(
                        self.grouping_tensor[p,idx,:] * self.network.T2[k,:,idx]
                    )
                    if neighbor_num:
                        self.payoff_tensor[p][idx][k] = np.dot(self.grouping_tensor[p,idx,:],
                            self.network.T2[k,:,idx]) * new_motivation[idx] / neighbor_num
                    else:
                        self.payoff_tensor[p][idx][k] = 0

    def cal_evaluation(self):
        """Calculate the evaluation of each population.
        """
        self.evaluation = []
        for p in range(self.P):
            self.evaluation.append(np.sum(self.payoff_tensor[p,:,:]))

    def cal_fitness(self):
        """Calculate the fitness of each solution.
        """
        self.fitness = []
        avg_fitness = sum(self.evaluation) / len(self.evaluation)
        for item in self.evaluation:
            self.fitness.append(item / avg_fitness)

    def selection(self):
        """Selection is applied to the current population
        to create an intermediate population.
        """
        self.intermediate_generation = []
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

    def recombination(self):
        """Implement recombination function to create the next generation.
        """
        next_generation_num = 0
        mid_generation_num = len(self.intermediate_generation)
        next_generation = np.empty([self.P, self.network.scale, self.network.scale])

        while next_generation_num < self.P:
            parent1 = self.intermediate_generation[next_generation_num % mid_generation_num]
            parent2 = self.intermediate_generation[
                (next_generation_num + 1) % mid_generation_num]

            if random.random() < self.pc:
                # crossover
                child1, child2 = self.crossover(parent1, parent2)
                next_generation[next_generation_num] = child1.copy()
                next_generation[next_generation_num+1] = child2.copy()

            else:
                # reproduction
                next_generation[next_generation_num] = parent1.copy()
                next_generation[next_generation_num+1] = parent2.copy()
                next_generation_num += 2

        self.grouping_tensor = next_generation.copy()

    def crossover(self, parent1, parent2):
        """Implement crossover function to create the next generation.

        Args:
            parent1 (ndarray): one solution in intermediate generation.
            parent2 (ndarray): one solution in intermediate generation.

        Returns:
            tuple: two offspring.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(self.network.scale):
            if sum(child1[i] * child2[i]) < 0.5 and random.random() < self.pc:
                # Connecting agents are completely different.
                # Row swap.
                child1[i], child2[i] = child2[i].copy(), child1[i].copy()
                # Column swap.
                child1[:,i], child2[:,i] = child2[:,i].copy(), child1[:,i].copy()

        return child1, child2

    def mutation(self):
        """Implement mutation function.
        """
        index = list(range(self.network.scale))
        for p in range(self.P):
            if random.random() < self.pm:
                i, j = random.sample(index, 2)
                if sum(self.grouping_tensor[p][i] * self.grouping_tensor[p][j]) < 0.5:
                    # Connecting agents are completely different.
                    # Row swap.
                    self.grouping_tensor[p][i], self.grouping_tensor[p][j] = \
                        self.grouping_tensor[p][j].copy(), self.grouping_tensor[p][i].copy()
                    # Column swap.
                    self.grouping_tensor[p,:,i], self.grouping_tensor[p,:,j] = \
                        self.grouping_tensor[p,:,j].copy(), self.grouping_tensor[p,:,i].copy()

    def write2file(self, path):
        """Write to disk file.

        Args:
            path (_type_): _description_
        """
        pass

        