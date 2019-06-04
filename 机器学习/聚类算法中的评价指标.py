# coding: utf-8
from collections import Counter
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def common_labels(samples, labels):
    '''
    :param samples: samples = np.array([ [ 0.94839057, -0.79470075],
    [-0.066221,   -1.48863894],
    [-0.52495579, -1.0756786 ],
    [ 1.48936899,  2.1204466 ],
    [ 1.88202617, 1.2000786 ]])
    :param labels: np.array([0, 0, 0, 1, 1])
    :return: {0: array([[ 0.94839057, -0.79470075],
       [-0.066221  , -1.48863894],
       [-0.52495579, -1.0756786 ]]), 1: array([[ 1.48936899,  2.1204466 ],
       [ 1.88202617,  1.2000786 ]])}
    '''
    dict = {}
    n_labels = [i for i in Counter(labels)]
    for i in n_labels:
        dict.update({i:samples[labels == i]})
    return dict

def cluster_in_out_set(target_sample, samples, labels):
    assert samples.shape[0] == len(labels)

    # pick_common_labels 是一个字典,键是标签,值是该标签对应的簇
    pick_common_labels = common_labels(samples, labels)

    for i in pick_common_labels:

        if target_sample in pick_common_labels[i]:
            target_label = i

    labels_samples = np.array([label for label in pick_common_labels])
    other_label = labels_samples[labels_samples != target_label]

    cluster_in_set = pick_common_labels[target_label]
    cluster_out_set = [
        pick_common_labels[i] for i in other_label
    ]
    
    return cluster_in_set, cluster_out_set

def Euclidean_Distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def Silhouette_Coefficient(samples, labels):
    # a_i = 1 / (C_i - 1)×d(i, j), j∈C_i, i ≠j
    # b_i = min(1/C_j×d(i, j)), i ≠j, j∈C_j
    # s_i = (b_i - a_i)÷max(b_i, a_i), if C_i
    # s = mean(s_i), i∈samples

    s = []
    for i in samples:
        #遍历所有样本
        '''
        1.取出该样本所在簇的所有集合
        2.求该样本与其他样本的欧几里得的均值
        i --> 当前样本
        cluster_in_set --> 该样本所在簇的集合 C_1
        cluster_out_set --> 其他簇的集合,[C_2, C_3,...]
        cluster_in_out_set --> 这个函数的功能是找出该样本所在簇的集合和其他簇的集合
        '''
        cluster_in_set, cluster_out_set = cluster_in_out_set(i, samples, labels) #cluster_in 本簇内的集合
        # cluster_in_other_set --> 簇内其他样本的集合
        cluster_in_other_set = cluster_in_set[cluster_in_set != i]
        # #################################
        # 簇内相似度
        # #################################
        a_i = np.mean([
            Euclidean_Distance(i, i_other) for i_other in cluster_in_other_set #i_other是任一簇内其他样本
        ])

        # #################################
        # 簇外不相似度
        # #################################
        b_i = np.min([
            np.mean([
                Euclidean_Distance(i, i_other) for i_other in cluster_out  #i_other其他簇任一样本
            ])    for cluster_out in cluster_out_set #cluster_out任一其他簇
        ])

        # #################################
        # 轮廓系数
        # #################################
        a = b_i - a_i
        b = max(a_i, b_i)
        s_i = a / b
        s.append(s_i)

    Silhouette = np.mean(s)
    return Silhouette


samples = np.array([ [ 0.94839057, -0.79470075],
    [-0.066221,   -1.48863894],
    [-0.52495579, -1.0756786 ],
    [ 1.48936899,  2.1204466 ],
    [ 1.88202617, 1.2000786 ],
    [ 2.517, 3.60786 ]
                     ])
labels = np.array([0, 2, 0, 1, 1, 2])
# print('cluster_in_set 输出的内容是:', cluster_in_set)
# print('cluster_in_set 的格式是:', cluster_in_set.shape)
# print('cluster_in_set 的类型是:', cluster_in_set.shape)
# print('cluster_out_set 输出的内容是:', cluster_out_set)
# print('cluster_out_set 的格式是:', len(cluster_out_set))
print('Silhouette_Coefficient输出的内容是:', Silhouette_Coefficient(samples, labels))
print('sklearn输出的内容是:', silhouette_score(samples, labels))

