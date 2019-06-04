'''
本代码利用numpy实现kmeans算法的原理，其中利用sklearn中的生成数据的包来生成数据
生成3类数据点，每个类别的数据服从高斯分布
'''
import numpy as np
from sklearn.datasets import make_blobs
import random
import matplotlib.pyplot as plt

def Gaussion(n, center):
    x, _ = make_blobs(n_samples=n, n_features=2, cluster_std=1.0,
                      centers=center, shuffle=False, random_state=42)

    return x


def train(n_samples, centers, k, iteration):

    # #######################################
    # 构造数据集，初始化质心左边，定义计数器
    # #######################################
    x = np.mat(Gaussion(n_samples, centers))
    c = np.mat(x[np.random.randint(1, n_samples + 1, size=k)])
    last_c = c
    n_samples = len(x)
    y = [np.random.randint(1, k + 1) for _ in range(n_samples)]
    count = 0
    difference = 100
    # #######################################
    # 开始定义循环条件
    # #######################################
    while count <= iteration and difference > 0.1:

        '''
        每个样本与每个质点的距离为x^2 -2 * x*c + c^2,然后对其开根号，因是否开根号对计算
        不影响，反而拖慢运行速度，故本代码不进行开根号计算d1、d2、d3分别对应上文提到的3个因子
        最终对其求和，即可得到任意样本到任意质心的距离，其中D.shape=[样本数， 质心数]
        '''
        count += 1
        d1 = np.sum(np.multiply(x, x), axis=1)
        d2 = -2 * x * c.T
        d3 = np.sum(np.multiply(c, c), axis=1)
        D = d1 + d2 + d3.T

        # #######################################
        # 得到样本-距离矩阵后，对其进行重新打标
        # #######################################
        D_min = np.argmin(D, axis=1)
        y = D_min.getA().reshape(-1)

        # #############################################
        # 得计算质心点
        # 首先取到该label下对应的样本，对其求均值，
        # 解出新的质心坐标，然后拼接在一起新的质心矩阵坐标
        # #############################################
        c0 =  np.mean([x[i]  for i in range(len(y)) if y[i] == 0], axis=0)
        c1 =  np.mean([x[i]  for i in range(len(y)) if y[i] == 1], axis=0)
        c2 =  np.mean([x[i]  for i in range(len(y)) if y[i] == 2], axis=0)
        c = np.vstack((c0, c1, c2))

        # #############################################
        # 计算两次更新后的质心坐标变化范围
        # #############################################
        difference = np.sum(np.square(c - last_c))
        print('epoch:{}, difference:{}'.format(count, difference))
        last_c = c

    print('计算完毕，质心点为：', c)


if __name__ == '__main__':

    # 参数配置
    n_samples = 500
    centers = [(-5, -5), (0, 0), (15, 15)]
    num = 1000
    k = 3

    # 开始训练数据，本数据集高斯分布
    train(n_samples, centers, k, num)
