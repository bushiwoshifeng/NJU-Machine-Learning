import numpy as np


def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。
    :该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''
    Gini = np.zeros((len(label), 3))
    res = np.float(0)
    for i in range(len(label)):
        if label[i] == 0:
            Gini[feature[i][index]][0] += 1
        else:
            Gini[feature[i][index]][1] += 1
    for i in range(len(label)):
        Gini[i][2] = Gini[i][0] + Gini[i][1]
        if Gini[i][2] != 0:
            Gini[i][0] /= Gini[i][2]
            Gini[i][1] /= Gini[i][2]
        Gini[i][2] /= len(label)
        res += Gini[i][2] * (1 - Gini[i][0]**2 - Gini[i][1]**2)
    return res


# test
feature = [[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]]
label = [0, 1, 0, 0, 1]
index = 0
print(calcGini(feature, label, index))
