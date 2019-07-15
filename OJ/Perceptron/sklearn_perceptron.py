#encoding=utf8
import os
from sklearn.linear_model.perceptron import Perceptron
import pandas as pd

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

# 获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

# 训练数据
clf = Perceptron(eta0=0.1, max_iter=500)
clf.fit(train_data, train_label)
res = clf.predict(test_data)

# 保存
res = {"result": res}
res = pd.DataFrame(res)
res.to_csv('./step2/result.csv', index=0)