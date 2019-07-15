#encoding=utf8
import numpy as np


# 计算样本间距离
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


# 计算质心
def cal_Cmass(data):
    '''
    input:data(ndarray):数据样本
    output:mass(ndarray):数据样本质心
    '''
    n, m = data.shape
    Cmass = np.zeros(m)
    Cmass = Cmass.reshape(1, -1)
    for i in range(n):
        Cmass += data[i, :]
    Cmass /= n
    return Cmass


# 计算每个样本到质心的距离，并按照从小到大的顺序排列
def sorted_list(data, Cmass):
    '''
    input:data(ndarray):数据样本
          Cmass(ndarray):数据样本质心
    output:dis_list(list):排好序的样本到质心距离
    '''
    dis_list = []
    n = data.shape[0]
    for i in range(n):
        dis_list.append(distance(data[i, :], Cmass))
    dis_list.sort()
    return dis_list


'''
[
    2.5504270456160367, 2.7970717409436814, 2.993145280935, 3.1328402051351536,
    3.1345854014971692, 3.200352819835331, 3.3464751222948097,
    3.7714714513579373, 5.015440818744084, 5.123737119303874
]
'''