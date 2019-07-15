#encoding=utf8
import numpy as np
import random


#寻找eps邻域内的点
def findNeighbor(j, X, eps):
    N = []
    for p in range(X.shape[0]):  #找到所有领域内对象
        temp = np.sqrt(np.sum(np.square(X[j] - X[p])))  #欧氏距离
        if (temp <= eps):
            N.append(p)
    return N


#dbscan算法
def dbscan(X, eps, min_Pts):
    '''
    input:X(ndarray):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    '''
    #********* Begin *********#
    k = -1
    NeighborPts = []  #array,某点领域内的对象
    Ner_NeighborPts = []
    fil = []  #初始时已访问对象列表为空
    gama = [x for x in range(len(X))]  #初始时将所有点标记为未访问
    cluster = [-1 for y in range(len(X))]
    while len(gama) > 0:
        j = random.choice(gama)
        gama.remove(j)  #未访问列表中移除
        fil.append(j)  #添加入访问列表
        NeighborPts = findNeighbor(j, X, eps)
        if len(NeighborPts) < min_Pts:
            cluster[j] = -1
        else:
            k = k + 1
            cluster[j] = k
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts = findNeighbor(i, X, eps)
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if (cluster[i] == -1):
                        cluster[i] = k
    #********* End *********#
    return cluster
'''
# dbscan算法
def dbscan(X, eps, min_Pts):
    '''
    input:X(ndarray):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    '''
    clusters = []
    m, n = X.shape  # (105, 2)
    # 初始化核心对象集合
    core = []
    for j in range(m):
        Neighbor_index = findNeighbor(j, X, eps)
        if len(Neighbor_index) >= min_Pts:
            core.append(j)
    # 初始化聚类簇数
    k = 0
    # 初始化未访问样本集合
    gamma = list(range(m))
    while len(core) > 0:
        gamma_old = gamma
        core_object = np.random.choice(core)
        # 初始化队列
        queue = []
        queue.append(core_object)
        gamma = list(set(gamma) ^ set(queue))
        while len(queue) > 0:
            q = queue.pop(0)
            Neighbor_index = findNeighbor(q, X, eps)
            if len(Neighbor_index) >= min_Pts:
                delta = list(set(gamma) & set(Neighbor_index))
                # add delta to queue
                for delta_i in delta:
                    queue.append(delta_i)
                gamma = list(set(gamma) ^ set(delta))
        C_k = list(set(gamma_old) ^ set(gamma))
        # add C_k
        cluster = []
        clusters.append(cluster)
        for i in range(len(C_k)):
            clusters[k].append(list(X[C_k[i]]))
        k += 1
        core = list(set(core) ^ set(C_k))
    return clusters
'''