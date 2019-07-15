#encoding=utf8
from sklearn.svm import LinearSVC


def linearsvc_predict(train_data, train_label, test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    '''
    clf = LinearSVC(C=1.4, max_iter=50000)
    clf.fit(train_data, train_label)
    predict = clf.predict(test_data)
    return predict