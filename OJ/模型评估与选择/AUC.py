import numpy as np


def calAUC(prob, labels):
    '''
    计算AUC并返回
    :param prob: 模型预测样本为Positive的概率列表，类型为ndarray
    :param labels: 样本的真实类别列表，其中1表示Positive，0表示Negtive，类型为ndarray
    :return: AUC，类型为float
    '''
    n = len(prob)
    AUC = np.float(0)
    a = np.ones((2, n))
    a[0, :] = prob
    a[1, :] = labels
    a = a.T[np.lexsort(a[::-1, :])].T   # *
    prob = a[0, :]
    labels = a[1, :]
    prob = prob.tolist()
    labels = labels.tolist()
    prob.reverse()
    labels.reverse()
    #统计m-(m1),m+(m2)
    prob = prob.reverse()
    m1 = np.int(0)
    m2 = np.int(0)

    for i in range(n):
        if labels[i] == 1:
            m2 = m2 + 1
        else:
            m1 = m1 + 1
    x1 = np.float(0)
    y1 = np.float(0)
    for i in range(n):
        if labels[i] == 1:
            x2 = x1
            y2 = y1 + 1 / m2
        else:
            x2 = x1 + 1 / m1
            y2 = y1
        AUC = AUC + 1 / 2 * (x2 - x1) * (y1 + y2)
        x1 = x2
        y1 = y2
    return AUC


prob = [0.1, 0.4, 0.3, 0.8]
labels = [0, 0, 1, 1]
print(calAUC(prob, labels))