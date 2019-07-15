import numpy as np


def mse_score(y_predict, y_test):
    '''
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    ouput:mse(float):mse损失函数值
    '''
    mse = np.mean((y_predict - y_test)**2)
    return mse


#r2
def r2_score(y_predict, y_test):
    '''
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    output:r2(float):r2值
    '''
    y_mean = np.mean(y_test)
    r2 = 1 - np.sum((y_predict - y_test)**2) / np.sum((y_mean - y_test)**2)
    return r2


class LinearRegression:
    def __init__(self):
        '''初始化线性回归模型'''
        self.theta = None

    def fit_normal(self, train_data, train_label):
        '''
        input:train_data(ndarray):训练样本
              train_label(ndarray):训练标签
        '''
        train_data = np.c_[train_data, np.ones(train_data.shape[0])]
        self.theta = np.dot(
            np.linalg.inv(np.dot(train_data.T, train_data)),
            np.dot(train_data.T, train_label))
        return self

    def predict(self, test_data):
        '''
        input:test_data(ndarray):测试样本
        '''
        test_data = np.c_[test_data, np.ones(test_data.shape[0])]
        y_predict = np.dot(test_data, self.theta)
        return y_predict