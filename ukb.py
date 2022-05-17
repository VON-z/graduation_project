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

# here put the third-party packages

# here put the local import source

class UKB():
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
        
    def __call__(self):
        pass
    