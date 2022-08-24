#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 LinYulong. All Rights Reserved 
#
# @Time    : 2022/6/12
# @Author  : LinYulong

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import find

coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                        [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                        [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                        [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                        [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                        [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                        [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                        [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                        [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
                        [1340.0, 725.0], [1740.0, 245.0]])


def get_dist_mat(coordinates):
    num = coordinates.shape[0]
    distmat = np.zeros((52, 52))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distmat


dist_mat = get_dist_mat(coordinates)
num_ant = 40  # 蚂蚁个数
num_city = coordinates.shape[0]  # 城市个数
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.1  # 信息素的挥发速度
Q = 1
iter = 0
iter_max = 250
eta_table = 1.0 / (dist_mat + np.diag([1e10] * num_city))  # 启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
pheromone_table = np.ones((num_city, num_city))  # 信息素矩阵
path_table = np.zeros((num_ant, num_city)).astype(int)  # 路径记录表
dist_mat = get_dist_mat(coordinates)  # 城市的距离矩阵
length_avg = np.zeros(iter_max)  # 各代路径的平均长度
length_best = np.zeros(iter_max)  # 各代及其之前遇到的最佳路径长度
path_best = np.zeros((iter_max, num_city))  # 各代及其之前遇到的最佳路径长度
while iter < iter_max:
    # 随机产生各个蚂蚁的起点城市
    if num_ant <= num_city:  # 城市数比蚂蚁数多
        path_table[:, 0] = np.random.permutation(range(0, num_city))[:num_ant]
    else:  # 蚂蚁数比城市数多，需要补足
        path_table[:num_city, 0] = np.random.permutation(range(0, num_city))[:]
        path_table[num_city:, 0] = np.random.permutation(range(0, num_city))[:num_ant - num_city]
    length = np.zeros(num_ant)  # 计算各个蚂蚁的路径距离
    for i in range(num_ant):
        visiting = path_table[i, 0]  # 当前所在的城市
        # visited = set() #已访问过的城市，防止重复
        # visited.add(visiting) #增加元素
        unvisited = set(range(num_city))  # 未访问的城市
        unvisited.remove(visiting)  # 删除元素
        for j in range(1, num_city):  # 循环 num_city-1 次，访问剩余的 num_city-1 个城市
            # 每次用轮盘法选择下一个要访问的城市
            list_unvisited: list = list(unvisited)
            probtrans = np.zeros(len(list_unvisited))
            for k in range(len(list_unvisited)):
                probtrans[k] = np.power(pheromone_table[visiting][list_unvisited[k]], alpha) \
                               * np.power(eta_table[visiting][list_unvisited[k]], alpha)
            cumsum_probtrans = (probtrans / sum(probtrans)).cumsum()
            cumsum_probtrans -= np.random.rand()
            k = list_unvisited[find(cumsum_probtrans > 0)[0]]  # 下一个要访问的城市
            path_table[i, j] = k
            unvisited.remove(k)
            # visited.add(k)
            length[i] += dist_mat[visiting][k]
            visiting = k
        length[i] += dist_mat[visiting][path_table[i, 0]]  # 蚂蚁的路径距离包括最后一个城市和第一个城市的距离
    # print length
    # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数
    length_avg[iter] = length.mean()
    if iter == 0:
        length_best[iter] = length.min()
        path_best[iter] = path_table[length.argmin()].copy()
    else:
        if length.min() > length_best[iter - 1]:
            length_best[iter] = length_best[iter - 1]
            path_best[iter] = path_best[iter - 1].copy()
        else:
            length_best[iter] = length.min()
            path_best[iter] = path_table[length.argmin()].copy()
    # 更新信息素
    change_pheromone_table = np.zeros((num_city, num_city))
    for i in range(num_ant):
        for j in range(num_city - 1):
            change_pheromone_table[path_table[i, j]][path_table[i, j + 1]] += Q / dist_mat[path_table[i, j]][
                path_table[i, j + 1]]
        change_pheromone_table[path_table[i, j + 1]][path_table[i, 0]] += Q / dist_mat[path_table[i, j + 1]][path_table[i, 0]]
    pheromone_table = (1 - rho) * pheromone_table + change_pheromone_table
    iter += 1  # 迭代次数指示器+1
    # 观察程序执行进度，该功能是非必须的
    if (iter - 1) % 20 == 0:
        print(iter - 1)
# 做出平均路径长度和最优路径长度
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
axes[0].plot(length_avg, 'k', marker=u'')
axes[0].set_title('Average Length')
axes[0].set_xlabel(u'iteration')
axes[1].plot(length_best, 'k', marker=u'')
axes[1].set_title('Best Length')
axes[1].set_xlabel(u'iteration')
fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
plt.close()
# 作出找到的最优路径图
best_path = path_best[-1]
plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker=u'$\cdot$')
plt.xlim([-100, 2000])
plt.ylim([-100, 1500])
for i in range(num_city - 1):  #
    m, n = best_path[i], best_path[i + 1]
    print(m, n)
    plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')
plt.plot([coordinates[best_path[0]][0], coordinates[n][0]], [coordinates[best_path[0]][1], coordinates[n][1]], 'b')
ax = plt.gca()
ax.set_title("Best Path")
ax.set_xlabel('X axis')
ax.set_ylabel('Y_axis')
plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
plt.close()
