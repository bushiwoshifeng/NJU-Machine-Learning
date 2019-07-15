from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def news_predict(train_sample, train_label, test_sample):
    '''
    训练模型并进行预测，返回预测结果
    :param train_sample:原始训练集中的新闻文本，类型为ndarray
    :param train_label:训练集中新闻文本对应的主题标签，类型为ndarray
    :param test_sample:原始测试集中的新闻文本，类型为ndarray
    :return 预测结果，类型为ndarray
    '''
    vec = CountVectorizer()
    train_sample = vec.fit_transform(train_sample)
    test_sample = vec.transform(test_sample)
    tfidf = TfidfTransformer()
    train_sample = tfidf.fit_transform(train_sample)
    test_sample = tfidf.transform(test_sample)
    clf = MultinomialNB(alpha=0.02)
    clf.fit(train_sample, train_label)
    res = clf.predict(test_sample)
    return res