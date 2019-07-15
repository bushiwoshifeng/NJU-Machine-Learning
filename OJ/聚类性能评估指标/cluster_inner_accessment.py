import numpy as np
from sklearn.metrics import davies_bouldin_score


def calc_DBI(feature, pred):
    '''
    计算并返回DB指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: DB指数
    '''
    k = len(list(set(pred)))
    n = pred.size
    u = np.zeros((k, feature.shape[1]))
    avg = np.zeros(k)
    cnt = np.zeros(k)
    # calc u
    for i in range(n):
        cnt[pred[i] - 1] += 1
        u[pred[i] - 1] += feature[i]
    for i in range(k):
        u[i, :] /= cnt[i]
    # calc avg(C_{i})
    for i in range(n):
        avg[pred[i] - 1] += np.sqrt(
            np.sum(np.square(feature[i, :] - u[pred[i] - 1, :])))
    for i in range(k):
        avg[i] /= cnt[i]
    # calc DBI
    _sum = 0
    for i in range(k):
        _max = 0
        for j in range(k):
            if i != j:
                d = np.sqrt(np.sum(np.square(u[i, :] - u[j, :])))
                if _max < (avg[i] + avg[j]) / d:
                    _max = (avg[i] + avg[j]) / d
        _sum += _max
    _sum /= k
    return _sum


def calc_DI(feature, pred):
    '''
    计算并返回Dunn指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: Dunn指数
    '''
    # calc diam, dmin
    n = pred.size
    k = len(list(set(pred)))
    diam = np.zeros(k)
    dmin = np.full((k, k), 1e7)  # init with big value
    for i in range(n - 1):
        for j in range(i + 1, n):
            if pred[i] == pred[j]:
                if np.sqrt(np.sum(
                        np.square(feature[i, :] -
                                  feature[j, :]))) > diam[pred[i] - 1]:
                    diam[pred[i] - 1] = np.sqrt(
                        np.sum(np.square(feature[i, :] - feature[j, :])))
            else:
                if np.sqrt(np.sum(
                        np.square(feature[i, :] -
                                  feature[j, :]))) < dmin[pred[i] -
                                                          1][pred[j] - 1]:
                    dmin[pred[j] -
                         1][pred[i] -
                            1] = dmin[pred[i] - 1][pred[j] - 1] = np.sqrt(
                                np.sum(np.square(feature[i, :] -
                                                 feature[j, :])))
    _diam = np.max(diam)
    _dmin = 1e7
    for i in range(k):
        for j in range(k):
            if dmin[i][j] < _dmin:
                _dmin = dmin[i][j]
    return _dmin / _diam


feature = np.array([[6, 4], [6, 9], [2, 3], [3, 4], [7, 10], [8, 11], [1, 0]])
pred = np.array([1, 2, 1, 1, 2, 2, 1])
print(calc_DBI(feature, pred))
print(davies_bouldin_score(feature, pred))
print(calc_DI(feature, pred))