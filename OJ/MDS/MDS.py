# -*- coding: utf-8 -*-
import numpy as np


def distances(data, m):
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist[i][j] = np.sqrt(np.sum(np.square(data[i] - data[j])))
    return dist


def mds(data, d):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    '''
    m, n = data.shape
    # 计算dist^2, dist^2i, dist^2j, dist^2ij
    dist = distances(data, m)  # 获得距离
    dist2ij = np.square(dist)
    dist2 = np.sum(dist2ij) / (m * m)
    dist2i = np.zeros(m)
    dist2j = np.zeros(m)
    for i in range(m):
        dist2i[i] = np.sum(dist2ij[i, :]) / m
        dist2j[i] = np.sum(dist2ij[:, i]) / m
    # 计算B
    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            B[i][j] = (dist2ij[i][j] - dist2i[i] - dist2j[j] + dist2) / (-2)
    # 矩阵分解得到特征值与特征向量
    value, vector = np.linalg.eigh(B)
    idx = np.argsort(value)[::-1]
    idx = idx[:d]
    gamma = np.sqrt(np.diag(value[idx]))
    V = vector[:, idx]
    # 计算Z
    Z = np.dot(V, gamma)
    return Z
