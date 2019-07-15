# -*- coding: utf-8 -*-
import numpy as np


def mds(data, d, k, Max):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    '''
    #********* Begin *********#
    #计算dist2,dist2i,dist2j,dist2ij
    m, n = data.shape
    dist = np.ones((m, m)) * Max
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m, m))
    for i in range(m):
        distance = np.power(np.tile(data[i], (m, 1)) - data, 2).sum(axis=1)
        index = np.argsort(distance)
        q = index[:k]
        for l in q:
            dist[i][l] = np.power(data[i] - data[l], 2).sum()
    for i in range(m):
        disti[i] = np.mean(dist[i, :])
        distj[i] = np.mean(dist[:, i])
    distij = np.mean(dist)
    #计算B
    for i in range(m):
        for j in range(m):
            B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
    #矩阵分解得到特征值与特征向量
    lamda, V = np.linalg.eigh(B)
    #计算Z
    index = np.argsort(-lamda)[:d]
    diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:d]))
    V_selected = V[:, index]
    Z = V_selected.dot(diag_lamda)
    #********* End *********#
    return Z


def isomap(data, d, k, Max=10000):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
          k(int):最近的k个样本
          Max(int):表示无穷大
    output:Z(ndarray):降维后的数据
    '''
    Z = mds(data, d, k, Max)
    '''
    # 计算dist2, dist2i, dist2j, dist2ij
    # 计算B
    # 矩阵分解得到特征值与特征向量
    # 计算Z
    '''
    return Z