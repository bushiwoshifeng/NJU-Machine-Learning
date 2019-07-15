from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

#获取训练数据
train_data = pd.read_csv('./step3/train_data.csv')
#获取训练标签
train_label = pd.read_csv('./step3/train_label.csv')
train_label = train_label['target']
#获取测试数据
test_data = pd.read_csv('./step3/test_data.csv')

#训练并预测
lr = LinearRegression()
lr.fit(train_data, train_label)
predict = lr.predict(test_data)

#保存数据
predict = {"result": predict}
predict = pd.DataFrame(predict)
predict.to_csv('./step3/result.csv', index=0)
