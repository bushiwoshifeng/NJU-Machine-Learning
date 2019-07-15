import numpy as np
import random
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []

    def fit(self, feature, label):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        def getBag(n, seed=0):
            if seed != 0:
                random.seed(seed)
            index = []
            for i in range(n):
                index.append(random.randint(0, n - 1))
            index = list(set(index))
            return index

        for i in range(self.n_model):
            index = getBag(feature.shape[0])
            lr = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=3)
            _feature = feature[index]
            _label = label[index]
            lr.fit(_feature, _label)
            self.models.append(lr)

    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        predict = []
        for i in range(feature.shape[0]):
            _predict = []
            _feature = np.reshape(feature[i], (1, -1))
            for j in range(len(self.models)):
                model = self.models[j]
                _predict.append(model.predict(_feature)[0])
            obj = Counter(_predict)
            predict.append(obj.most_common(1)[0][0])
        return np.array(predict)