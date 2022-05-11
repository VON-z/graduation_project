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

# here put the third-party packages

# here put the local import source

if __name__ == '__main__':
    pass

# # Draw evaluation figure.
# # 支持中文以及负数
# plt.rcParams['font.sans-serif']=['STSong']  # SimHei黑体 STSong宋体
# plt.rcParams['axes.unicode_minus'] = False

# x = list(range(IT))
# y = [sum(evaluation[i]) / POPULATION for i in range(IT)]
# plt.subplot(1, 2, 1)
# plt.plot(x, y, linewidth=2.0)
# # plt.show()

# # Draw heatmap.
# plt.subplot(1, 2, 2)
# sns.set_theme()
# sns.heatmap(evaluation.T)
# plt.show()
# print('Breakpoint')