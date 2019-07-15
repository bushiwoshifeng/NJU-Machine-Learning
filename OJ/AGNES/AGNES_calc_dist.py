import numpy as np


def calc_min_dist(cluster1, cluster2):
    '''
    计算簇间最小距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最小距离
    '''
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]
    dis = 1e7
    for i in range(n1):
        for j in range(n2):
            if np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j]))) < dis:
                dis = np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j])))
    return dis


def calc_max_dist(cluster1, cluster2):
    '''
    计算簇间最大距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最大距离
    '''
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]
    dis = 0
    for i in range(n1):
        for j in range(n2):
            if np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j]))) > dis:
                dis = np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j])))
    return dis


def calc_avg_dist(cluster1, cluster2):
    '''
    计算簇间平均距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的平均距离
    '''
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]
    dis = 0
    for i in range(n1):
        for j in range(n2):
            dis += np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j])))
    dis = dis / (n1 * n2)
    return dis


cluster1 = np.array([[0, 1, 0], [1, 0, 1], [1, 2, 3.2], [0, 0, 1.2],
                     [1, 1, 0.1]])
cluster2 = np.array([[10.1, 20.3, 9], [8.2, 15.3, 11]])
print(calc_min_dist(cluster1, cluster2))
print(calc_max_dist(cluster1, cluster2))
print(calc_avg_dist(cluster1, cluster2))