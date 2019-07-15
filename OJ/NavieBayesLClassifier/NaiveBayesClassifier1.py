import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''
        row = label.shape[0]
        col = feature.shape[1]
        cnt0 = cnt1 = 0
        # init
        self.condition_prob[0] = {}
        self.condition_prob[1] = {}
        for i in range(col):
            self.condition_prob[0][i] = {}
            self.condition_prob[1][i] = {}
            for j in range(1, 4):  # 1, 2, 3
                self.condition_prob[0][i][j] = 0
                self.condition_prob[1][i][j] = 0
        # compute
        for i in range(row):
            if label[i] == 0:
                cnt0 += 1
            else:
                cnt1 += 1
            for j in range(col):
                self.condition_prob[label[i]][j][feature[i][j]] += 1
        self.label_prob[0] = cnt0 / row
        self.label_prob[1] = cnt1 / row
        for j in range(col):
            for k in range(1, 4):
                self.condition_prob[0][j][k] /= cnt0
                self.condition_prob[1][j][k] /= cnt1

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        row = feature.shape[0]
        col = feature.shape[1]
        res = np.zeros(row)
        for i in range(row):
            p0 = self.label_prob[0]
            p1 = self.label_prob[1]
            for j in range(col):
                p0 *= self.condition_prob[0][j][feature[i][j]]
                p1 *= self.condition_prob[1][j][feature[i][j]]
            if p0 > p1:
                res[i] = 0
            else:
                res[i] = 1
        return res