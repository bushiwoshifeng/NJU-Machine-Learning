import numpy as np

# g(D,A)=H(D)−H(D,A)


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。
     该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''
    n = len(label)
    cnt = np.zeros((n, 3))
    cnt0 = 0
    cnt1 = 0
    for i in range(n):
        if label[i] == 0:
            cnt[feature[i][index], 0] = cnt[feature[i][index], 0] + 1
            cnt0 = cnt0 + 1
        else:
            cnt[feature[i][index], 1] = cnt[feature[i][index], 1] + 1
            cnt1 = cnt1 + 1

    for i in range(n):
        cnt[i, 2] = cnt[i, 0] + cnt[i, 1]
        if cnt[i, 2] != 0:
            cnt[i, 0] /= cnt[i, 2]
            cnt[i, 1] /= cnt[i, 2]

    # compute H_D
    p_0 = cnt0 / n
    p_1 = cnt1 / n
    H_D = -p_0 * np.log2(p_0) - p_1 * np.log2(p_1)

    # compute H_DA
    H_DA = np.float(0)
    for i in range(n):
        # H_DA += (cnt[i, 2] * (
        #     -cnt[i, 0] * np.log2(cnt[i, 0]) - cnt[i, 1] * np.log2(cnt[i, 1])))
        if cnt[i, 0] != 0:
            H_DA += cnt[i, 2] / n * (-cnt[i, 0] * np.log2(cnt[i, 0]))
        if cnt[i, 1] != 0:
            H_DA += cnt[i, 2] / n * (-cnt[i, 1] * np.log2(cnt[i, 1]))
    res = H_D - H_DA
    return res