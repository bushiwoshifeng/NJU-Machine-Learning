#encoding=utf8
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def ada_classifier(train_data, train_label, test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试标签
    output:predict(ndarray):预测结果
    '''
    adaboost = AdaBoostClassifier(learning_rate=1.2)
    adaboost.fit(train_data, train_label)
    predict = adaboost.predict(test_data)
    return predict