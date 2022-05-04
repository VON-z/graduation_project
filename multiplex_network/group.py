# # -*- encoding: utf-8 -*-
# '''
# @File    :   group.py
# @Time    :   2022/03/21 14:37:04
# @Author  :   VONz
# @Version :   1.0
# @Contact :   1036991178@qq.com
# @License :   (C)Copyright 2021-2022, VONz.
# @Desc    :   This file implements functions related to group.
# '''

# # here put the standard library

# # here put the third-party packages
# import numpy as np

# # here put the local import source


# class Group():
#     """Define a single group.
#     """
#     layers_num = 0 # the number of network layers.
#     max_member_num = 0 # the maximum number of group members.
#     adjacency_matrix = None # the adjacency matrix of whole network.
#     motivation_network = None # the adjacency matrix of motivation network.

#     def __init__(self, index, **kw) -> None:
#         self.index = index
#         self.member_num = 0
#         self.agents = []
#         self.payoff = 0

#         if 'agents' in kw:
#             self.agents = kw['agents'].copy()

#         self.cal_group_motivation_network()
#         self.cal_group_adjacency_matrix()

#     def add_agent(self, a):
#         """Add an agent.

#         Args:
#             a (Agent): Agent instance to be added.
#         """
#         if self.member_num < Group.max_member_num:
#             self.agents.append(a)
#             self.member_num += 1

#     def cal_group_motivation_network(self):
#         """Extract the group's motivation network.
#         """
#         member_num = len(self.agents)
#         self.group_motivation_network = np.empty(
#             [Group.layers_num, member_num, member_num], dtype = float)
#         for i, a1 in enumerate(self.agents):
#             for j, a2 in enumerate(self.agents):
#                 self.group_motivation_network[:,i,j] = Group.motivation_matrix[:,a1.index,a2.index]

#     def cal_group_adjacency_matrix(self):
#         """Extract the group's adjacency matrix.
#         """
#         member_num = len(self.agents)
#         self.group_adjacency_matrix = np.empty(
#             [Group.layers_num, member_num, member_num], dtype = float) 
#         for i, a1 in enumerate(self.agents):
#             for j, a2 in enumerate(self.agents):
#                 self.group_adjacency_matrix[:,i,j] = Group.adjacency_matrix[:,a1.index,a2.index]

#     def cal_new_motivation(self):
#         """Calculate new motivation of this group.
#         """

#     def cal_payoff(self):
#         """Calculate the payoff of this group.
#         """

                
