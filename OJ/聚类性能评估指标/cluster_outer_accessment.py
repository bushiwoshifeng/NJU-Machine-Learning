import numpy as np


def calc_JC(y_true, y_pred):
    '''
    计算并返回JC系数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: JC系数
    '''
    a = 0
    b = 0
    c = 0
    n = y_true.size
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                a += 1
            if y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                b += 1
            if y_true[i] != y_true[j] and y_pred[i] == y_pred[j]:
                c += 1
    return a / (a + b + c)


def calc_FM(y_true, y_pred):
    '''
    计算并返回FM指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: FM指数
    '''
    a = 0
    b = 0
    c = 0
    n = y_true.size
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                a += 1
            if y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                b += 1
            if y_true[i] != y_true[j] and y_pred[i] == y_pred[j]:
                c += 1
    return np.sqrt(a / (a + b) * a / (a + c))


def calc_Rand(y_true, y_pred):
    '''
    计算并返回Rand指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: Rand指数
    '''
    a = 0
    d = 0
    m = y_true.size
    for i in range(m - 1):
        for j in range(i + 1, m):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                a += 1
            if y_true[i] != y_true[j] and y_pred[i] != y_pred[j]:
                d += 1
    return 2 * (a + d) / (m * (m - 1))


y_true = np.array([0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 2, 2])
print(calc_JC(y_true, y_pred))
print(calc_FM(y_true, y_pred))
print(calc_Rand(y_true, y_pred))