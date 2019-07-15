# -*- coding: utf-8 -*-
from sklearn.manifold import LocallyLinearEmbedding


def lle(data, d, k):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维度
          k(int):邻域内样本数
    output:Z(ndarray):降维后数据
    '''
    lle = LocallyLinearEmbedding(n_components=d, n_neighbors=k)
    Z = lle.fit_transform(data)
    return Z
