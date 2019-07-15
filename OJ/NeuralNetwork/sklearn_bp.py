#encoding=utf8
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

#获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
#获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
#获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

mlp = MLPClassifier(
    solver='lbfgs', max_iter=100, alpha=1e-5)  #使用默认hidden_layer_sizes
mlp.fit(train_data, train_label)
res = mlp.predict(test_data)
res = {"result": res}
res = pd.DataFrame(res)
res.to_csv('./step2/result.csv', index=0)