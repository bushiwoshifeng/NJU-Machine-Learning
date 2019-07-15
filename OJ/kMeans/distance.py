#encoding=utf8
import numpy as np


def distance(x, y, p=2):
    '''
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离，等于2时为欧氏距离
    output:distance(float):x到y的距离      
    '''
    res = 0
    if p == 1:
        res = np.sum(np.abs(x - y))
    else:
        res = np.sqrt(np.sum(np.square(x - y)))
    return res


x = np.array([1, 2])
y = np.array([2, 3])
print(distance(x, y, 1))
print(distance(x, y, 2))