from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''
    scaler = StandardScaler()
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.fit_transform(test_feature)
    clf = KNeighborsClassifier(algorithm='auto',
                               leaf_size=30,
                               metric='minkowski',
                               metric_params=None,
                               n_jobs=1,
                               n_neighbors=1,
                               p=2,
                               weights='uniform')
    clf.fit(train_feature, train_label)
    predict = clf.predict(test_feature)
    return predict