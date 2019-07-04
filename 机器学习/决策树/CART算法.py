
import numpy as np
import matplotlib.pyplot as plt



# make_classification生成三元分类模型数据
from sklearn.datasets import make_classification

# 关键参数有n_samples（生成样本数）， n_features（样本特征数）， n_redundant（冗余特征数）和n_classes（输出的类别数）
# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X1, Y1 = make_classification(n_samples=400, n_features=5, n_redundant=0, n_clusters_per_class=1, n_classes=2)
plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=3, marker='o')
plt.show()