import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
''' 
KNN算法:
   功能:输入具有m个特征的样本,即为m维向量,得到一个类别值
   实现方法:
       1.遍历所有样本中的点,求每个点与检测点的距离值(欧几里得距离)
       2.按照大小进行排序,
       3.取前N(奇数)个点,统计属于哪个类别的数量最多即为最佳点
    代码撰写思路:
       1.定义计算欧几里得距离的函数,
       2.遍历样本中的所有点,将结果储存到字典中
       3.按照大小进行排序,取前N个
'''


def Euclidean_Distance(vector1, vector2):
    assert vector1.shape == vector2.shape, 'vector1和vector2的形状必须一致'

    return np.linalg.norm(vector1 - vector2)

# names = ['sepal length', 'sepal_width', 'petal_length', 'petal_width', 'cla']

# class KNN:
#     pass
def KNN(data, N):
    df = pd.read_csv(data, sep=',')
    df.replace('Iris-setosa', 1, inplace=True)
    df.replace('Iris-virginica', 2, inplace=True)
    df.replace('Iris-versicolor', 3, inplace=True)

    # 对其进行向量化
    X = df.iloc[:, 0:-1]
    Y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    x_test = np.mat(x_test)
    y_test = np.mat(y_test)
    x_train = np.mat(x_train)
    y_pred = []
    # 取出每个待检测样本
    for j in range(x_test.shape[0]):
        target = x_test[j]
        count = {}


        # 将每个待检测模板与已有数据中所有样本求欧式距离,并储存到字典中
        for i in range(x_train.shape[0]):
            
            if y_train.iloc[i] == 1:
                count.update({str(Euclidean_Distance(x_train[i], target)):1})
            elif y_train.iloc[i] == 2:
                count.update({str(Euclidean_Distance(x_train[i], target)):2})
            elif y_train.iloc[i] == 3:
                count.update({str(Euclidean_Distance(x_train[i], target)):3})
    
        '''
        依次索引排序后的值,按从小到大去字典 里面搜索,取掉类别值,储存到列表中,然后选取个数最多的值
        '''
        classtic = []
        for i in range(N):
            classtic.append(count[sorted(count)[i]])
        result = Counter(classtic).most_common(1)[0][0]

        # 将计算得到最大
        y_pred.append(abs(result - y_test[:,j]))
    pro = 0
    for _ in y_pred:
        if _ == 0:
            pro += 1
    print('总样本个数是:{}, 当N取{}时,预测正确的样本个数为:{}, 正确率为:{}'.format(len(y_pred), N, pro,pro/len(y_pred)))
    return pro/len(y_pred)

if __name__ == '__main__':

    data_path = r'../数据集/iris.data'

    # 对比不同的N值下,分类结果如何
    N = np.arange(1, 21, 2)
    Y = [KNN(data_path, i) for i in N]
    plt.plot(N, Y, 'r')
    plt.show()
