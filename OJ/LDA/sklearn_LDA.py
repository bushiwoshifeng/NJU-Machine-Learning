#encoding=utf8 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda(x,y):
    '''
    input:x(ndarray):待处理数据
          y(ndarray):待处理数据标签
    output:x_new(ndarray):降维后数据
    '''
    #********* Begin *********#
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x,y)
    x_new = lda.transform(x)
    #********* End *********#
    return x_new