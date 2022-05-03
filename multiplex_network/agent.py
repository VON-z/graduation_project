# -*- encoding: utf-8 -*-
'''
@File    :   agents.py
@Time    :   2022/03/21 13:23:27
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   This file implements functions related to individual modeling.
'''

# here put the standard library
import random

# here put the third-party packages

# here put the local import source

class Agent():
    """Single individual modeling.
    Including self index, index of belonging group, skill list.
    """
    skill_num = 0
    def __init__(self, index, **kw) -> None:
        """Initialize the agent.

        Args:
            index (int): self index.
            groupindex (int): group index.
            motivation (float): learning efficiency. (0-1)
            optional:
                skills (list): skills list.
        """
        self.index = index
        self.groupindex = None
        self.motivation = 0
        self.skills = []

        if 'skills' in kw:
            # Data initialization.
            for item in kw['skills']:
                self.skills.append(item)
        else:
            # Random initialization.
            for _ in range(Agent.skill_num):
                self.skills.append(random.betavariate(4, 4))
