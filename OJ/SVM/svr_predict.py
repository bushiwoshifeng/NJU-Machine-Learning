#encoding=utf8
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

'''
# 数据集
from sklearn.datasets import load_boston
boston = load_boston()
x = boston['data']
y = boston['target']
# 划分训练集与测试机
train_data,test_data,train_label,test_label = train_test_split(x,y,test_size=0.2,random_state=995)
'''

def svr_predict(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    '''
#     svr = SVR()
#     svr.fit(train_data, train_label)
#     predict = svr.predict(test_data)
    rfr = RandomForestRegressor()
    rfr.fit(train_data, train_label)
    predict = rfr.predict(test_data)
    return predict