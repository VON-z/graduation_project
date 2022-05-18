# -*- encoding: utf-8 -*-
'''
@File    :   ukb.py
@Time    :   2022/05/16 22:11:01
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   None
'''

# here put the standard library
import math
import random
# here put the third-party packages
import numpy as np
from tqdm import trange
# here put the local import source

class UKB():
    """Uniform K-means Based Algorithm.
    """
    def __init__(self, network, group_scale, skill_num, k) -> None:
        """Initialization.

        Args:
            network (Network): network model.
            group_scale (int): max member number in a group.
            skill_num (int): skill number.
            k (int): cluster num.
        """
        self.network = network
        self.group_scale = group_scale
        self.skill_num = skill_num
        self.k = k

        self.group_num = math.ceil(self.network.scale / self.k)
        self.data = []
        for a in self.network.agents:
            self.data.append(a.skills.copy())
        self.data = np.array(self.data)

        self.centers = self.data[:self.k]
        self.new_centers = None
        self.classifications = np.zeros(self.network.scale, dtype=int)
        self.cluster_member_num = None

        self.grouping_matrix = np.zeros([self.network.scale, self.network.scale], dtype=int)
        self.payoff_matrix = None
        self.evaluation = 0

    def assignment(self):
        """assignment agents.
        """
        self.classifications.fill(-1)
        self.cluster_member_num = np.zeros(self.k)
        for i, a in enumerate(self.data):
            dist = []
            for c in self.centers:
                dist.append(np.linalg.norm(a-c))
            while self.classifications[i] == -1:
                cluster_idx = dist.index(min(dist))
                if self.cluster_member_num[cluster_idx] < self.group_num:
                    self.classifications[i] = cluster_idx
                    self.cluster_member_num[cluster_idx] += 1
                else:
                    dist[cluster_idx] = 999

    def cal_new_centers(self):
        """calculate new centers.
        """
        cluster = []
        self.new_centers = []
        for _ in range(self.k):
            cluster.append([])
        for i, cluster_idx in enumerate(self.classifications):
            cluster[cluster_idx].append(self.data[i])

        for cluster_idx in range(self.k):
            c = np.array(cluster[cluster_idx])
            center = c.mean(axis=0)
            self.new_centers.append(center)
        self.new_centers = np.array(self.new_centers)

    def grouping(self):
        """grouping.
        """
        grouping = []
        for _ in range(self.group_num):
            grouping.append([])

        cluster = []
        for _ in range(self.k):
            cluster.append([])
        for i, cluster_idx in enumerate(self.classifications):
            cluster[cluster_idx].append(i)

        for cluster_idx in range(self.k):
            rank = list(range(len(cluster[cluster_idx])))
            random.shuffle(rank)
            for i, group_idx in enumerate(rank):
                grouping[group_idx].append(cluster[cluster_idx][i])

        for group in grouping:
            for a1 in group:
                for a2 in group:
                    if not a1 == a2:
                        self.grouping_matrix[a1][a2] = 1

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

    def __call__(self, limit=500):
        """_summary_

        Args:
            limit (int, optional): _description_. Defaults to 500.

        Returns:
            tuple: _description_
        """
        # clustering.
        for _ in trange(limit):
            self.assignment()
            self.cal_new_centers()
            if (self.new_centers == self.centers).all():
                break
            else:
                self.centers = self.new_centers.copy()

        # grouping.
        self.grouping()
        self.cal_payoff()
        self.cal_evaluation()
        return self.grouping_matrix, self.evaluation
