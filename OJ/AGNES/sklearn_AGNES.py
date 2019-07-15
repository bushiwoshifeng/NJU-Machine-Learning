#encoding=utf8
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def Agglomerative_cluster(data):
    '''
    对红酒数据进行聚类
    :param data: 数据集，类型为ndarray
    :return: 聚类结果，类型为ndarray
    '''
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    agnes = AgglomerativeClustering(n_clusters=3)
    result = agnes.fit_predict(data)
    return result