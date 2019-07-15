#encoding=utf8
from sklearn.cluster import DBSCAN


def data_cluster(data):
    '''
    input: data(ndarray) :数据
    output: result(ndarray):聚类结果
    '''
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    result = dbscan.fit_predict(data)
    return result