from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture
'''
图像的基础操作
from PIL import Image

# 读取road.jpg到im变量中
im = Image.open('road.jpg')

# 将im转换成ndarray
img = np.array(im)

# 将image变形为[-1, 3]的shape, 并保存至img_reshape
img_reshape = img.reshape(-1, 3)

# pred为聚类算法的预测结果，将簇为0的点设置成红色，簇为1的点设置成蓝色
img[pred == 0, :] = [255, 0, 0]
img[pred == 1, :] = [0, 0, 255]

# 将img转换成Image类型
im = Image.fromarray(img.astype('uint8'))

# 将im保存为road.jpg
im.save('new_road.jpg')

'''

# 实际效果 https://www.educoder.net/tasks/63l9hq78gftw

im = Image.open('./step3/image/test.jpg')
img = np.array(im)  # (450, 600, 3)
img_reshape = img.reshape(-1, 3)  # (270000, 3)
gmm = GaussianMixture(n_components=3)
gmm.fit(img_reshape)
pred = gmm.predict(img_reshape)  # (270000, )
img_reshape[pred == 0, :] = [255, 255, 0]
img_reshape[pred == 1, :] = [0, 0, 255]
img_reshape[pred == 2, :] = [0, 255, 0]
img = img_reshape.reshape((450, 600, 3))
im = Image.fromarray(img.astype('uint8'))
im.save('./step3/dump/result.jpg')
