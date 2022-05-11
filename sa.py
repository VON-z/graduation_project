# -*- encoding: utf-8 -*-
'''
@File    :   sa.py
@Time    :   2022/05/10 19:17:22
@Author  :   VONz 
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   Implement simulated annealing to solve the problem.
'''

# here put the standard library
import math
import random
import os

# here put the third-party packages
import numpy as np
import scipy.linalg
from tqdm import trange

# here put the local import source

class SA():
    """Implement simulated annealing.
    """
    def __init__(self, network, group_scale, skill_num) -> None:
        """Initialization.

        Args:
            network (Network): network model.
            group_scale (int): max member number in a group.
            skill_num (int): skill number.
        """
        self.network = network
        self.group_scale = group_scale
        self.skill_num = skill_num

        self.block_diagonal_matrix = None
        self.generate_block_diagonal_matrix()

        self.grouping_matrix = None # The matrix representing the grouping result。
        self.payoff_matrix = None   # The matrix records payoff of the grouping result.
        self.evaluation = 0

        # __call__
        self.C_max = None
        self.T = None
        self.alpha = None
        self.L = None
        self.best_solution = None
        self.best_solution_evaluation = None

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

    def generate_initial_solution(self):
        """Generate initial solution.
        """
        self.grouping_matrix = self.block_diagonal_matrix.copy()
        for i in range(self.network.scale):
            j = random.randint(0, self.network.scale-1)
            if sum(self.grouping_matrix[i] * self.grouping_matrix[j]) < 0.5:
                # Connecting agents are completely different.
                # Row swap.
                self.grouping_matrix[i], self.grouping_matrix[j] = \
                    self.grouping_matrix[j].copy(), self.grouping_matrix[i].copy()
                # Column swap.
                self.grouping_matrix[:,i], self.grouping_matrix[:,j] = \
                    self.grouping_matrix[:,j].copy(), self.grouping_matrix[:,i].copy()

    def cal_payoff(self):
        """Calculate payoff for each agent in each skill.
        """
        self.payoff_matrix = np.empty([self.network.scale, self.skill_num])
        new_motivation = []
        # Calculate motivation after grouping.
        for idx in range(self.network.scale):
            neighbor_num = np.count_nonzero(
                self.grouping_matrix[idx,:] * self.network.T1[:, idx]
            )
            if neighbor_num:
                new_motivation.append(
                    self.network.agents[idx].motivation + np.dot(
                        self.grouping_matrix[idx,:], self.network.T1[:, idx]) / neighbor_num
                )
            else:
                new_motivation.append(self.network.agents[idx].motivation)

        # Calculate skill promotion.
        for idx in range(self.network.scale):
            for k in range(self.skill_num):
                neighbor_num = np.count_nonzero(
                    self.grouping_matrix[idx,:] * self.network.T2[k,:,idx]
                )
                if neighbor_num:
                    self.payoff_matrix[idx][k] = np.dot(self.grouping_matrix[idx,:],
                        self.network.T2[k,:,idx]) * new_motivation[idx] / neighbor_num
                else:
                    self.payoff_matrix[idx][k] = 0

    def cal_evaluation(self):
        """Calculate the evaluation of the solution.
        """
        self.evaluation = np.sum(self.payoff_matrix)

    def generate_new_solution(self):
        """Generate solution.
        """
        index = list(range(self.network.scale))
        flag = True
        while flag:
            i, j = random.sample(index, 2)
            if sum(self.grouping_matrix[i] * self.grouping_matrix[j]) < 0.5:
                # Connecting agents are completely different.
                # Row swap.
                self.grouping_matrix[i], self.grouping_matrix[j] = \
                    self.grouping_matrix[j].copy(), self.grouping_matrix[i].copy()
                # Column swap.
                self.grouping_matrix[:,i], self.grouping_matrix[:,j] = \
                    self.grouping_matrix[:,j].copy(), self.grouping_matrix[:,i].copy()
                flag = False

    def __call__(self, C_max, T, alpha, L):
        """Execute the algorithm.

        Args:
            C_max (int): Iteration rounds.
            T (int): initial temperature.
            alpha (float): attenuation factor.
            L (int): the number of iterations L for each value of T.

        Returns:
            evaluation (list): the evaluation value of candidate solution in each temperature.
            self.best_solution (ndarray): best grouping matrix.
            self.best_solution_evaluation (float): best grouping evaluation.
        """
        self.C_max = C_max
        self.T = T
        self.alpha = alpha
        self.L = L

        self.generate_initial_solution()
        self.cal_payoff()
        self.cal_evaluation()
        self.best_solution = self.grouping_matrix.copy()
        self.best_solution_evaluation = self.evaluation

        evaluation = []
        evaluation.append(self.evaluation)
        for i in trange(self.C_max):
            original_solution = self.grouping_matrix.copy()
            for k in range(self.L):
                # Store the current grouping result, payoff matrix and evaluation.
                current_solution = self.grouping_matrix.copy()
                current_payoff = self.payoff_matrix.copy()
                current_evaluation = self.evaluation

                # Generate a new solution
                self.generate_new_solution()
                self.generate_initial_solution()
                self.cal_payoff()
                self.cal_evaluation()

                delta_e = self.evaluation - current_evaluation
                if not random.random() < math.exp(min(0, delta_e) / self.T) :
                    self.grouping_matrix = current_solution.copy()
                    self.payoff_matrix = current_payoff.copy()
                    self.evaluation = current_evaluation
                evaluation.append(self.evaluation)
                # Record the best grouping result.
                if self.evaluation > self.best_solution_evaluation:
                    self.best_solution_evaluation = self.evaluation
                    self.best_solution = self.grouping_matrix.copy()
            # If the solution does not change after L rounds, then exit.
            if (original_solution == self.grouping_matrix).all():
                break
            self.T *= self.alpha

        return evaluation.copy(), self.best_solution, self.best_solution_evaluation
