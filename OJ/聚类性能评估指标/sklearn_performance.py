from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def classification_performance(y_true, y_pred, y_prob):
    '''
    返回准确度、精准率、召回率、f1 Score和AUC
    :param y_true:样本的真实类别，类型为`ndarray`
    :param y_pred:模型预测出的类别，类型为`ndarray`
    :param y_prob:模型预测样本为`Positive`的概率，类型为`ndarray`
    :return:
    '''
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return acc, precision, recall, f1, roc_auc


y_prob = [0.1, 0.4, 0.3, 0.8]
y_pred = [1, 0, 1, 1]
y_true = [0, 0, 1, 1]
print(classification_performance(y_true, y_pred, y_prob))