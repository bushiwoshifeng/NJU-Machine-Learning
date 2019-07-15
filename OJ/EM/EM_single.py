import numpy as np
from scipy import stats


def em_single(init_values, observations):
    """
    模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率
    :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param observations:抛掷硬币的实验结果记录，类型为list。
    :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """
    res = np.zeros(len(init_values))
    cnt1 = np.zeros((len(init_values), 2))
    for i in range(len(observations)):
        temp = observations[i]
        prob = np.ones(len(init_values))
        cnt2 = 0
        for k in range(len(temp)):
            if temp[k] == 1:
                cnt2 += 1
            for j in range(len(init_values)):
                if temp[k] == 1:
                    prob[j] *= init_values[j]
                else:
                    prob[j] *= (1 - init_values[j])
        _sum = sum(prob)
        prob /= _sum
        for j in range(len(init_values)):
            cnt1[j][0] += prob[j] * cnt2
            cnt1[j][1] += prob[j] * (len(temp) - cnt2)
    for i in range(len(init_values)):
        res[i] = cnt1[i][0] / (cnt1[i][0] + cnt1[i][1])
    return res


init_values = [0.2, 0.7]
observations = [[1, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1], [0, 1, 1, 0, 0]]
print(em_single(init_values, observations))