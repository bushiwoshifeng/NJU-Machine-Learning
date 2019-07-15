# -*- coding: utf-8 -*-
from sklearn.manifold import Isomap


def isomap(data, d, k):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维度
          k(int):最近的k个样本
    output:Z(ndarray):降维后数据
    '''
    isomap = Isomap(n_components=d, n_neighbors=k)
    Z = isomap.fit_transform(data)
    return Z
