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
import numpy as np
# here put the local import source
from data import load_experiment_data

# Hyperparameters
SKILL_NUM = 5
MAX_MEMBER_NUM = 5

# Network Hyperparameters
NETWORK_SCALE = [20, 40, 60, 80, 100]
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
    plt.figure(figsize=(7, 7))

    ax = plt.subplot()
    cu = 1.8
    ax.spines['left'].set_linewidth(cu)
    ax.spines['right'].set_linewidth(cu)
    ax.spines['bottom'].set_linewidth(cu)
    ax.spines['top'].set_linewidth(cu)

    plt.yticks(fontproperties='Times New Roman', size=24)
    plt.xticks(fontproperties='Times New Roman', size=24)

    x = list(range(len(y)))
    plt.plot(x, y, linewidth=2.0)
    plt.savefig(path)

def draw_evaluation_line(r, alg, network_type, network_scale, **kw):
    """ER network.

    Args:
        r (int): _description_
        alg (str): _description_
        network_type (str): _description_
        network_scale (int): _description_
    """
    evaluation = None
    # Genetic Algorithm.
    if alg == 'GA':
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

    # Simulated Annealing.
    if alg == 'SA':
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

def draw_ga_chart(labels, data, p):
    """_summary_

    Args:
        labels (list): _description_
        data (ndarray): _description_
        p (float):
    """
    plt.figure(figsize=(9, 9))
    x = np.arange(len(labels))  # the label locations
    width = 0.20    # the width of the bars
    cu = 1.8
    ax = plt.subplot() 
    ax.spines['left'].set_linewidth(cu)
    ax.spines['right'].set_linewidth(cu)
    ax.spines['bottom'].set_linewidth(cu)
    ax.spines['top'].set_linewidth(cu)

    rects = ax.bar(x - 0.5*width, data[0], width-0.02, label='GAc', color='#5b9bd5')
    ax.bar_label(rects, padding=3, fmt='%.1f', fontproperties='Times New Roman', fontsize=18)
    rects = ax.bar(x + 0.5*width, data[1], width-0.02, label='GAn', color='#70ad47')
    ax.bar_label(rects, padding=3, fmt='%.1f', fontproperties='Times New Roman', fontsize=18)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('平均收益')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.legend()

    plt.tight_layout()
    path = os.path.join('./figure', 'crossover')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'er_p{}.png'.format(p)))

def draw_performance(labels, data, **kw):
    """_summary_

    Args:
        labels (list): _description_
        data (ndarray): _description_
    """
    plt.figure(figsize=(7, 7))
    x = np.arange(len(labels))  # the label locations
    width = 0.20    # the width of the bars
    cu = 1.8
    ax = plt.subplot() 
    ax.spines['left'].set_linewidth(cu)
    ax.spines['right'].set_linewidth(cu)
    ax.spines['bottom'].set_linewidth(cu)
    ax.spines['top'].set_linewidth(cu)

    rects = ax.bar(x - width, data[0], width-0.02, label='SA', color='#5b9bd5')
    ax.bar_label(rects, padding=3, fmt='%.1f', fontproperties='Times New Roman', fontsize=18)
    rects = ax.bar(x, data[1], width-0.02, label='GA', color='#70ad47')
    ax.bar_label(rects, padding=3, fmt='%.1f', fontproperties='Times New Roman', fontsize=18)
    rects = ax.bar(x + width, data[2], width-0.02, label='UKB', color='#ffc000')
    ax.bar_label(rects, padding=3, fmt='%.1f', fontproperties='Times New Roman', fontsize=18)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('平均收益')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize=20)

    plt.tight_layout()
    path = os.path.join('./figure', 'performance')
    if not os.path.exists(path):
        os.makedirs(path)
    if kw['network_type'] == 'ER':
        plt.savefig(os.path.join(path, 'er_p{}.png'.format(kw['p'])))
    if kw['network_type'] == 'WS':
        plt.savefig(os.path.join(path, 'ws_p{}.png'.format(kw['p'])))
    if kw['network_type'] == 'BA':
        plt.savefig(os.path.join(path, 'ba_m{}.png'.format(kw['m'])))

if __name__ == '__main__':
    # 支持中文以及负数
    plt.rcParams['font.sans-serif']=['STSong']  # SimHei黑体 STSong宋体
    plt.rcParams['axes.unicode_minus'] = False

    # 1.Draw evaluation line chart.
    ## Genetic Algorithm and Simulated Annealing.
    for alg in ['SA', 'GA']:
        ### ER
        for p in ER_P:
            draw_evaluation_line(r=0, alg=alg, network_type='ER',
            network_scale=60, p=p)

    # # 2.Draw evaluation bar chart. (GA crossover)
    # ## load data.
    # for p in [0.1, 0.5]:
    #     data_c = np.zeros([10, 5])
    #     data_n = np.zeros([10, 5])
    #     for r in range(10):
    #         for i, network_scale in enumerate(NETWORK_SCALE):
    #             _, _, data_c[r][i] = load_experiment_data(r, 'GA', 'ER', network_scale,
    #                 p=p, pc=0.5, pm=0.1)
    #             ### load no crossover data.
    #             path = os.path.join('./result(no_crossover)', str(r), 'GA', 'ER',
    #                 'er_n{}_p{}_pc{}_pm{}_bse.npy'.format(network_scale, p, 0.5, 0.1))
    #             data_n[r][i] = np.load(path)
    #     data_c_mean = data_c.mean(axis=0).reshape(1,5)
    #     data_n_mean = data_n.mean(axis=0).reshape(1,5)
    #     data = np.concatenate((data_c_mean, data_n_mean), axis=0)
    #     labels = ['20', '40', '60', '80', '100']
    #     draw_ga_chart(labels, data, p)

    # # 3.Draw performance bar.
    # ## ER
    # for p in ER_P:
    #     data_sa = np.zeros([10, 5])
    #     data_ga = np.zeros([10, 5])
    #     data_ukb = np.zeros([10, 5])
    #     ### load data.
    #     for r in range(10):
    #         for i, network_scale in enumerate(NETWORK_SCALE):
    #             #### SA
    #             _, _, data_sa[r][i] = load_experiment_data(r, 'SA', 'ER', network_scale,
    #                 p=p, t=100, alpha=0.97)
    #             #### GA
    #             _, _, data_ga[r][i] = load_experiment_data(r, 'GA', 'ER', network_scale,
    #                 p=p, pc=0.5, pm=0.1)
    #             #### UKB
    #             path = os.path.join('./result', str(r), 'UKB', 'ER',
    #                 'er_n{}_p{}_eva.npy'.format(network_scale, p))
    #             data_ukb[r][i] = np.load(path)
    #     data_sa_mean = data_sa.mean(axis=0).reshape(1,5)
    #     data_ga_mean = data_ga.mean(axis=0).reshape(1,5)
    #     data_ukb_mean = data_ukb.mean(axis=0).reshape(1,5)
    #     data = np.concatenate((data_sa_mean, data_ga_mean, data_ukb_mean), axis=0)
    #     labels = ['20', '40', '60', '80', '100']
    #     draw_performance(labels, data, network_type='ER',p=p)
    # ## WS
    # for p in WS_P:
    #     data_sa = np.zeros([10, 5])
    #     data_ga = np.zeros([10, 5])
    #     data_ukb = np.zeros([10, 5])
    #     ### load data.
    #     for r in range(10):
    #         for i, network_scale in enumerate(NETWORK_SCALE):
    #             #### SA
    #             _, _, data_sa[r][i] = load_experiment_data(r, 'SA', 'WS', network_scale,
    #                 p=p, t=100, alpha=0.97)
    #             #### GA
    #             _, _, data_ga[r][i] = load_experiment_data(r, 'GA', 'WS', network_scale,
    #                 p=p, pc=0.5, pm=0.1)
    #             #### UKB
    #             path = os.path.join('./result', str(r), 'UKB', 'WS',
    #                 'ws_n{}_p{}_eva.npy'.format(network_scale, p))
    #             data_ukb[r][i] = np.load(path)
    #     data_sa_mean = data_sa.mean(axis=0).reshape(1,5)
    #     data_ga_mean = data_ga.mean(axis=0).reshape(1,5)
    #     data_ukb_mean = data_ukb.mean(axis=0).reshape(1,5)
    #     data = np.concatenate((data_sa_mean, data_ga_mean, data_ukb_mean), axis=0)
    #     labels = ['20', '40', '60', '80', '100']
    #     draw_performance(labels, data, network_type='WS',p=p)
    # ## BA
    # for m in BA_M:
    #     data_sa = np.zeros([10, 5])
    #     data_ga = np.zeros([10, 5])
    #     data_ukb = np.zeros([10, 5])
    #     ### load data.
    #     for r in range(10):
    #         for i, network_scale in enumerate(NETWORK_SCALE):
    #             #### SA
    #             _, _, data_sa[r][i] = load_experiment_data(r, 'SA', 'BA', network_scale,
    #                 m=m, t=100, alpha=0.97)
    #             #### GA
    #             _, _, data_ga[r][i] = load_experiment_data(r, 'GA', 'BA', network_scale,
    #                 m=m, pc=0.5, pm=0.1)
    #             #### UKB
    #             path = os.path.join('./result', str(r), 'UKB', 'BA',
    #                 'ba_n{}_m{}_eva.npy'.format(network_scale, m))
    #             data_ukb[r][i] = np.load(path)
    #     data_sa_mean = data_sa.mean(axis=0).reshape(1,5)
    #     data_ga_mean = data_ga.mean(axis=0).reshape(1,5)
    #     data_ukb_mean = data_ukb.mean(axis=0).reshape(1,5)
    #     data = np.concatenate((data_sa_mean, data_ga_mean, data_ukb_mean), axis=0)
    #     labels = ['20', '40', '60', '80', '100']
    #     draw_performance(labels, data, network_type='BA',m=m)

# # Draw heatmap.
# plt.subplot(1, 2, 2)
# sns.set_theme()
# sns.heatmap(evaluation.T)
# plt.show()
# print('Breakpoint')
