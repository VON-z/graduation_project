# -*- encoding: utf-8 -*-
'''
@File    :   draw.py
@Time    :   2022/05/06 15:14:02
@Author  :   VONz
@Version :   1.0
@Contact :   1036991178@qq.com
@License :   (C)Copyright 2021-2022, VONz.
@Desc    :   Drawing kit.
'''

# here put the standard library
import os
# here put the third-party packages
import matplotlib.pyplot as plt
import seaborn as sns
# here put the local import source
from data import load_experiment_data

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5

# Network Hyperparameters
NETWORK_SCALE = [100]
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

def plot_line_chart(y, path):
    """_summary_

    Args:
        y (list): _description_
        path (str): _description_
    """
    x = list(range(len(y)))
    plt.plot(x, y, linewidth=2.0)
    plt.savefig(path)

def draw_evaluation_line(r, alg, network_type, network_scale, **kw):
    """_summary_

    Args:
        r (int): _description_
        alg (str): _description_
        network_type (str): _description_
        network_scale (int): _description_
    """
    evaluation = None
    # Genetic Algorithm.
    if alg == 'GA':
        ## ER
        if network_type == 'ER':
            for pc in PC:
                for pm in PM:
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        p=kw['p'], pc=pc, pm=pm)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    y = [sum(evaluation[i]) / POPULATION for i in range(C_MAX)]
                    plot_line_chart(y, os.path.join(path,
                        'er_n{}_p{}_pc{}_pm{}.png'.format(network_scale, kw['p'], pc, pm)))

        ## WS
        elif network_type == 'WS':
            for pc in PC:
                for pm in PM:
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        p=kw['p'], pc=pc, pm=pm)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    y = [sum(evaluation[i]) / POPULATION for i in range(C_MAX)]
                    plot_line_chart(y, os.path.join(path,
                        'ws_n{}_p{}_pc{}_pm{}.png'.format(network_scale, kw['p'], pc, pm)))
        ## BA
        elif network_type == 'BA':
            for pc in PC:
                for pm in PM:
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        m=kw['m'], pc=pc, pm=pm)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    y = [sum(evaluation[i]) / POPULATION for i in range(C_MAX)]
                    plot_line_chart(y, os.path.join(path,
                        'ba_n{}_m{}_pc{}_pm{}.png'.format(network_scale, kw['m'], pc, pm)))

    # Simulated Annealing.
    if alg == 'SA':
        ## ER
        if network_type == 'ER':
            for t in TE:
                for alpha in ALPHA:
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        p=kw['p'], t=t, alpha=alpha)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    plot_line_chart(evaluation, os.path.join(path,
                        'er_n{}_p{}_t{}_alpha{}.png'.format(network_scale, kw['p'], t, alpha)))

        ## WS
        elif network_type == 'WS':
            for t in TE:
                for alpha in ALPHA:                 
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        p=kw['p'], t=t, alpha=alpha)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    plot_line_chart(evaluation, os.path.join(path,
                        'ws_n{}_p{}_t{}_alpha{}.png'.format(network_scale, kw['p'], t, alpha)))
        ## BA
        elif network_type == 'BA':
            for t in TE:
                for alpha in ALPHA:               
                    evaluation, _, _ = \
                        load_experiment_data(r, alg, network_type, network_scale, 
                        m=kw['m'], t=t, alpha=alpha)
                    ### draw
                    path = os.path.join('./figure', str(r), alg, network_type)
                    if path:
                        if not os.path.exists(path):
                            os.makedirs(path)
                    plot_line_chart(evaluation, os.path.join(path,
                        'ba_n{}_m{}_t{}_alpha{}.png'.format(network_scale, kw['m'], t, alpha)))

def draw_evaluation_bar():
    pass

if __name__ == '__main__':
    # 支持中文以及负数
    plt.rcParams['font.sans-serif']=['STSong']  # SimHei黑体 STSong宋体
    plt.rcParams['axes.unicode_minus'] = False

    # Draw evaluation line chart.
    r=0
    ## Genetic Algorithm and Simulated Annealing.
    for alg in ['GA', 'SA']:
        for network_scale in NETWORK_SCALE:
            ### ER
            for p in ER_P:
                draw_evaluation_line(r=r, alg=alg, network_type='ER',
                network_scale=network_scale, p=p)

            # ### WS
            # for p in WS_P:
            #     draw_evaluation_line(r=r, alg=alg, network_type='WS',
            #     network_scale=network_scale, p=p)

            # ### BA
            # for m in BA_M:
            #     draw_evaluation_line(r=r, alg=alg, network_type='BA',
            #     network_scale=network_scale, m=m)

    # Draw evaluation bar chart.
    ## Genetic Algorithm and Simulated Annealing.

# # Draw heatmap.
# plt.subplot(1, 2, 2)
# sns.set_theme()
# sns.heatmap(evaluation.T)
# plt.show()
# print('Breakpoint')
