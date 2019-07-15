import numpy as np
from numpy.linalg import inv


def lda(X, y):
    '''
    input:X(ndarray):待处理数据
          y(ndarray):待处理数据标签，标签分别为0和1
    output:X_new(ndarray):处理后的数据
    '''
    #********* Begin *********#
    #划分出第一类样本与第二类样本
    x0 = []
    x1 = []
    for i in range(X.shape[0]):
    	if y[i]==0:
        	x0.append(X[i])
        else:
            x1.append(X[i])
    x0 = np.array(x0)
    x1 = np.array(x1)
    #获取第一类样本与第二类样本中心点
    u0 = x0.mean(axis=0)
    u1 = x1.mean(axis=0)
    #计算第一类样本与第二类样本协方差矩阵
    for i in range(x0.shape[0]):
        x0[i] -= u0
    for i in range(x1.shape[0]):
        x1[i] -= u1
    sigma0 = np.dot(x0.T, x0)
    sigma1 = np.dot(x1.T, x1)
    #计算类内散度矩阵
    S_w = sigma0 + sigma1
    #计算w
    w = np.dot(inv(S_w), (u0 - u1).T)
    #计算新样本集
    X_new = np.dot(X, w)
    #********* End *********#
    return X_new