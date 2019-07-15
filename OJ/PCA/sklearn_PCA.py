from sklearn.decomposition import PCA
from sklearn.svm import SVC


def cancer_predict(train_sample, train_label, test_sample):
    '''
    使用PCA降维，并进行分类，最后将分类结果返回
    :param train_sample:训练样本, 类型为ndarray
    :param train_label:训练标签, 类型为ndarray
    :param test_sample:测试样本, 类型为ndarray
    :return: 分类结果
    '''
    pca = PCA(n_components=15)
    train_sample = pca.fit_transform(train_sample)
    test_sample = pca.transform(test_sample)
    clf = SVC(C=1.0, kernel='linear')
    clf.fit(train_sample, train_label)
    result = clf.predict(test_sample)
    return result