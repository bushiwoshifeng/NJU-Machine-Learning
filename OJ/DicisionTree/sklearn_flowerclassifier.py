#encoding=utf8
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 读取数据
train_df = pd.read_csv('./step7/train_data.csv').as_matrix()
label = pd.read_csv('./step7/train_label.csv').as_matrix()
test_df = pd.read_csv('./step7/test_data.csv').as_matrix()

clf = tree.DecisionTreeClassifier()
clf.fit(train_df, label)
res = clf.predict(test_df)

# 保存
res = {'target': res}
res = pd.DataFrame(res)
res.to_csv('./step7/predict.csv', index=0)