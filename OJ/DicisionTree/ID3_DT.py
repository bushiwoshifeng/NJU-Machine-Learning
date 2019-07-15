import numpy as np
import pandas as pd
from collections import Counter
import math


class Node:
    def __init__(self, x=None, label=None, y=None, data=None):
        self.label = label  # label:子节点分类依据的特征
        self.x = x  # x:特征
        self.child = []  # child:子节点
        self.y = y  # y:类标记（叶节点才有）
        self.data = data  # data:包含数据（叶节点才有）

    def append(self, node):  # 添加子节点
        self.child.append(node)

    def predict(self, features):  # 预测数据所述类
        if self.y is not None:
            return self.y
        for c in self.child:
            if c.x == features[self.label]:
                return c.predict(features)


class DecisionTree(object):
    def __init__(self, epsilon=0, alpha=0):  # 预剪枝、后剪枝参数
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    def prob(self, datasets):  # 求概率
        datalen = len(datasets)
        labelx = set(datasets)
        p = {l: 0 for l in labelx}
        for d in datasets:
            p[d] += 1
        for i in p.items():
            p[i[0]] /= datalen
        return p

    def calc_ent(self, datasets):  # 求熵
        p = self.prob(datasets)
        ent = sum([-v * math.log(v, 2) for v in p.values()])
        return ent

    def cond_ent(self, datasets, col):  # 求条件熵
        labelx = set(datasets.iloc[col])
        p = {x: [] for x in labelx}
        for i, d in enumerate(datasets.iloc[-1]):
            p[datasets.iloc[col][i]].append(d)
        return sum([
            self.prob(datasets.iloc[col])[k] * self.calc_ent(p[k])
            for k in p.keys()
        ])

    def info_gain_train(self, datasets, datalabels):  # 求信息增益（互信息）
        datasets = datasets.T
        ent = self.calc_ent(datasets.iloc[-1])
        gainmax = {}
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets, i)
            gainmax[ent - cond] = i
        m = max(gainmax.keys())
        return gainmax[m], m

    def train(self, datasets, node):
        labely = datasets.columns[-1]
        if len(datasets[labely].value_counts()) == 1:
            node.data = datasets[labely]
            node.y = datasets[labely][0]
            return
        if len(datasets.columns[:-1]) == 0:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        gainmaxi, gainmax = self.info_gain_train(datasets, datasets.columns)
        #print('选择特征：', gainmaxi)
        if gainmax <= self.epsilon:  # 若信息增益（互信息）为0意为输入特征x完全相同而标签y相反
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return

        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        for Di in vc.index:
            node.label = gainmaxi
            child = Node(Di)
            node.append(child)
            new_datasets = pd.DataFrame(
                [list(i) for i in datasets.values if i[gainmaxi] == Di],
                columns=datasets.columns)
            self.train(new_datasets, child)

    def fit(self, datasets, labely):
        datasets = np.c_[datasets, labely]
        datasets = pd.DataFrame(datasets)
        self.train(datasets, self.tree)

    def predict(self, datasets):
        predict = []
        for i in range(datasets.shape[0]):
            predict.append(self.tree.predict(datasets[i]))
        return np.array(predict)