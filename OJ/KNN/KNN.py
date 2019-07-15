#encoding=utf8
import numpy as np


# k-Nearest Neighbor
class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None

    # 计算一个样本与数据集中所有样本的欧氏距离的平方
    def euclidean_distance(self, one_sample, X):
        one_sample = one_sample.reshape(1, -1)
        distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X,
                             2).sum(axis=1)
        return distances

    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''
        self.train_feature = feature
        self.train_label = label
        return

    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''
        n_samples, n_features = feature.shape
        predict = []
        for i in range(n_samples):
            labels = []  # k-neighbors' label
            distances = self.euclidean_distance(feature[i], self.train_feature)
            for distance in np.sort(distances)[:self.k]:
                label = self.train_label[distances == distance]
                labels.append(label[0])
            # 投票
            vote = {}
            labels = np.array(labels)
            for label in labels:
                if label not in vote:
                    vote[label] = 1
                else:
                    vote[label] += 1
            vote = sorted(vote.items(), key=lambda x: x[1], reverse=True)
            predict.append(vote[0][0])
        return predict