import numpy as np
from sklearn.decomposition import PCA


def pca(data, k):
    '''
    对data进行PCA，并将结果返回
    :param data:数据集，类型为ndarray
    :param k:想要降成几维，类型为int
    :return: 降维后的数据，类型为ndarray
    '''
    n, m = data.shape
    # 计算样本均值
    u = np.mean(data, axis=0)
    # demean
    after_demean = data - u
    # 计算 after_demean 的协方差矩阵 = (1/m)*(X^T*X)
    cov = np.cov(after_demean.T)
    # 计算协方差矩阵的特征值和特征向量
    value, vector = np.linalg.eig(cov)
    # 特征值排序，从大到小，因为特征值大的方差越大
    idx = np.argsort(vlaue)[::-1]  # 注意[::-1]这个写法
    idx = idx[:k]
    # 特征值最大的k个特征向量组成映射矩阵P
    P = vector[:, idx]
    # 降维
    return np.dot(after_demean, P)  # take care of this