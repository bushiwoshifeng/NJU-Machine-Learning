import numpy as np
from scipy import stats


def em_single(init_values, observations):
    """
    模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率。请不要修改！！
    :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param observations:抛掷硬币的实验结果记录，类型为list。
    :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """
    observations = np.array(observations)
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = init_values[0]
    theta_B = init_values[1]
    # E step
    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads
        # 两个二项分布
        contribution_A = stats.binom.pmf(num_heads, len_observation, theta_A)
        contribution_B = stats.binom.pmf(num_heads, len_observation, theta_B)
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        # 更新在当前参数下A、B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails
    # M step
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A, new_theta_B]


def em(observations, thetas, tol=1e-4, iterations=100):
    """
    模拟抛掷硬币实验并使用EM算法估计硬币A与硬币B正面朝上的概率。
    :param observations: 抛掷硬币的实验结果记录，类型为list。
    :param thetas: 硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param tol: 差异容忍度，即当EM算法估计出来的参数theta不怎么变化时，可以提前挑出循环。例如容忍度为1e-4，则表示若这次迭代的估计结果与上一次迭代的估计结果之间的L1距离小于1e-4则跳出循环。为了正确的评测，请不要修改该值。
    :param iterations: EM算法的最大迭代次数。为了正确的评测，请不要修改该值。
    :return: 将估计出来的硬币A和硬币B正面朝上的概率组成list或者ndarray返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """
    for _ in range(iterations):
        error = []
        new_theta = em_single(thetas, observations)
        for i in range(len(thetas)):
            error.append(new_theta[i] - thetas[i]) 
        if np.sum(np.abs(error)) < tol:
            break
        thetas = new_theta
    return new_theta


init_values = [0.2, 0.7]
observations = [[1, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1], [0, 1, 1, 0, 0]]
print(em(observations, init_values))