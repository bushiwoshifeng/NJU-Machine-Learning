import numpy as np
from scipy.stats import multivariate_normal


# 高斯混合聚类
class GMM(object):
    def __init__(self, n_components, max_iter=100):
        '''
        构造函数
        :param n_components: 想要划分成几个簇，类型为int
        :param max_iter: EM的最大迭代次数
        '''
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, train_data):
        '''
        训练，将模型参数分别保存至self.alpha，self.mu，self.sigma中
        :param train_data: 训练数据集，类型为ndarray
        :return: 无返回
        '''
        row, col = train_data.shape
        # 初始化每个高斯分布的响应系数
        self.alpha = np.array([1.0 / self.n_components] * self.n_components)
        # 初始化每个高斯分布的均值向量
        self.mu = np.random.rand(self.n_components, col)
        # 初始化每个高斯分布的协方差矩阵
        self.sigma = np.array([np.eye(col)] * self.n_components)
        # 正则化项
        self.regu = 1e-4 * np.eye(col)
        # 初始化r_{ji}
        r = np.zeros((row, self.n_components))
        for _ in range(self.max_iter):
            # calc r_{ji}
            for j in range(row):
                _sum = 0
                for k in range(self.n_components):
                    r[j][k] = self.alpha[k] * multivariate_normal(
                        mean=self.mu[k], cov=(self.sigma[k] + self.regu)).pdf(
                            train_data[j])
                    _sum += self.alpha[k] * multivariate_normal(
                        mean=self.mu[k], cov=(self.sigma[k] + self.regu)).pdf(
                            train_data[j])
                r[j] /= _sum
            # update mu, sigma, alpha
            for i in range(self.n_components):
                mu_1 = np.zeros(col)
                sigma1 = np.zeros((col, col))
                sum_r = np.sum(r[:, i])
                for j in range(row):
                    mu_1 += r[j][i] * train_data[j]
                    sigma1 += r[j][i] * np.dot(train_data[j] - self.mu[i],
                                               (train_data[j] - self.mu[i]).T)
                self.mu[i] = mu_1 / sum_r
                self.sigma[i] = sigma1 / sum_r
                self.alpha[i] = sum_r / row

    def predict(self, test_data):
        '''
        预测，根据训练好的模型参数将test_data进行划分。
        注意：划分的标签的取值范围为[0,self.n_components-1]，即若self.n_components为3，则划分的标签的可能取值为0,1,2。
        :param test_data: 测试集数据，类型为ndarray
        :return: 划分结果，类型为ndarray
        '''
        row, col = test_data.shape
        predict = []
        r = np.zeros((row, self.n_components))
        for j in range(row):
            _sum = 0
            for i in range(self.n_components):
                r[j][i] = self.alpha[i] * multivariate_normal(
                    mean=self.mu[i], cov=(self.sigma[i] + self.regu)).pdf(
                        test_data[j])
                _sum += self.alpha[i] * multivariate_normal(
                    mean=self.mu[i], cov=(self.sigma[i] + self.regu)).pdf(
                        test_data[j])
            r[j] /= _sum
            predict.append(np.argmax(r[j]))
        return np.array(predict)


'''
train_data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2],
                       [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
                       [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
                       [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
                       [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
                       [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
                       [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1]])
gmm = GMM(n_components=5)
gmm.fit(train_data)
'''

'''
import numpy as np
from scipy.stats import multivariate_normal


class GMM(object):
    def __init__(self, n_components, max_iter=100):
        '''
        构造函数
        :param n_components: 想要划分成几个簇，类型为int
        :param max_iter: EM的最大迭代次数
        '''
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, train_data):
        '''
        训练，将模型参数分别保存至self.alpha，self.mu，self.sigma中
        :param train_data: 训练数据集，类型为ndarray
        :return: 无返回
        '''
        row, col = train_data.shape
        # 初始化每个高斯分布的响应系数
        self.alpha = np.array([1.0 / self.n_components] * self.n_components)
        # 初始化每个高斯分布的均值向量
        self.mu = np.random.rand(self.n_components, col)
        # 初始化每个高斯分布的协方差矩阵
        self.sigma = np.array([np.eye(col)] * self.n_components)
        for j in range(self.max_iter):
            # e-step
            # 响应度矩阵，行对应样本，列对应响应度
            gamma = np.zeros((row, self.n_components))
            prob = np.zeros((row, self.n_components))
            # 计算各高斯分布中所有样本出现的概率，行对应样本，列对应高斯分布
            for k in range(self.n_components):
                prob[:, k] = multivariate_normal(
                    mean=self.mu[k], cov=self.sigma[k]).pdf(train_data)
            for k in range(self.n_components):
                gamma[:, k] = self.alpha[k] * prob[:, k]
            for i in range(row):
                gamma[i, :] /= np.sum(gamma[i, :])
            # m-step
            for k in range(self.n_components):
                # 第k个模型对所有样本的响应度之和
                Nk = np.sum(gamma[:, k])
                # 更新 mu
                # 对每个特征求均值
                self.mu[k, :] = np.sum(train_data * gamma[:, k].reshape(-1, 1),
                                       axis=0) / Nk
                # 更新 cov
                cov_k = np.matmul((train_data - self.mu[k]).T,
                                  (train_data - self.mu[k]) *
                                  gamma[:, k].reshape(-1, 1)) / Nk
                self.sigma[k, :, :] = cov_k
                # 更新 alpha
                self.alpha[k] = Nk / row

    def predict(self, test_data):
        '''
        预测，根据训练好的模型参数将test_data进行划分。
        注意：划分的标签的取值范围为[0,self.n_components-1]，即若self.n_components为3，则划分的标签的可能取值为0,1,2。
        :param test_data: 测试集数据，类型为ndarray
        :return: 划分结果，类型为你ndarray
        '''
        gamma = np.zeros((len(test_data), self.n_components))
        prob = np.zeros((len(test_data), self.n_components))
        for k in range(self.n_components):
            prob[:, k] = multivariate_normal(mean=self.mu[k],
                                             cov=self.sigma[k]).pdf(test_data)
        for k in range(self.n_components):
            gamma[:, k] = self.alpha[k] * prob[:, k]
        for i in range(len(test_data)):
            gamma[i, :] /= np.sum(gamma[i, :])
        return gamma.argmax(axis=1).flatten()
'''