#encoding=utf8
from sklearn.svm import SVC

#获取并处理鸢尾花数据
'''
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    #将标签为0的数据标签改为-1
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]
'''


def svc_predict(train_data, train_label, test_data, kernel):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          kernel(str):使用核函数类型:
              'linear':线性核函数
              'poly':多项式核函数
              'rbf':径像核函数/高斯核
    output:predict(ndarray):测试集预测标签
    '''
    clf = SVC(C=1.0,kernel=kernel)
    clf.fit(train_data, train_label)
    predict = clf.predict(test_data)
    return predict