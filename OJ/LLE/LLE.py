#encoding=utf8
import numpy as np


def distances(data, m, i, k):
    dist = np.zeros(m)
    for j in range(m):
        dist[j] = np.sqrt(np.sum(np.square(data[i] - data[j])))
    Q_i = np.argsort(dist)[:k]
    return Q_i


def lle(data, d, k):
    '''
    input:data(ndarray):待降维数据,行数为样本个数，列数为特征数
          d(int):降维后数据维数
          k(int):最近的k个样本
    output:Z(ndarray):降维后的数据
    '''
    m, n = data.shape
    W = np.zeros((m, m))
    for i in range(m):
        C = np.zeros((k, k))
        C_ls = 0
        # 确定样本 i 的邻域
        Q_i = distances(data, m, i, k)
        # 求矩阵 C 及其逆
        for j in range(k):
            for p in range(k):
                C[j][p] = np.sum(
                    (data[i] - data[Q_i[j]]) * (data[i] - data[Q_i[p]]))
        C_inv = np.linalg.inv(C + 1e-5 * np.eye(k))
        C_ls = np.sum(C_inv)
        # 求 W
        for j in range(k):
            C_jk = 0
            for k in range(k):
                C_jk += C_inv[j][k]
            W[i][Q_i[j]] = C_jk / C_ls
    # 求得 M 并矩阵分解
    M = np.dot((np.eye(m) - W).T, (np.eye(m) - W))
    value, vector = np.linalg.eigh(M)
    # 求 Z
    Z = vector[:, 1:1 + d]
    return Z
