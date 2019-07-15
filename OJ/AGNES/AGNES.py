import numpy as np


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


def AGNES(feature, k):
    '''
    AGNES聚类并返回聚类结果，量化距离时请使用簇间最大欧氏距离
    假设数据集为`[[1, 2], [10, 11], [1, 3]]，那么聚类结果可能为`[[1, 2], [1, 3]], [[10, 11]]]
    :param feature:数据集，类型为ndarray
    :param k:表示想要将数据聚成`k`类，类型为`int`
    :return:聚类结果，类型为list
    '''
    n_samples, n_features = feature.shape
    # 初始化簇
    clusters = [[] for _ in range(n_samples)]
    for i in range(n_samples):
        clusters[i].append(list(feature[i]))
    # AGNES
    q = len(clusters)
    while q > k:
        # 寻找距离最小的两个簇
        min_dis = 1e7
        a = b = -1
        for i in range(q - 1):
            for j in range(i + 1, q):
                if calc_max_dist(np.array(clusters[i]), np.array(
                        clusters[j])) < min_dis:
                    min_dis = calc_max_dist(np.array(clusters[i]),
                                            np.array(clusters[j]))
                    a = i
                    b = j
        # 合并
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
        q = len(clusters)

    return clusters
