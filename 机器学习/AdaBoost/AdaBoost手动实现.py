import numpy as np
from behold import Behold
import pysnooper
np.set_printoptions(precision=5)
# 符号函数
def sign(x, thresold=0.0):

    if x <= thresold:
        return 1.0
    else:
        return -1.0

def sign_reverse(x, thresold=0.0):

    if x <= thresold:
        return -1
    else:
        return 1

def bool2num(bool_str):
    if bool_str:
        return 0
    else:
        return 1

def errror_rate(y_true, y_pred, w):

    error = []
    for i, j in zip(y_true, y_pred):

        if i == j:

            error.append(0)
        else:
            error.append(1)
    error_num = np.sum(error * w)
    return error_num

# 计算分类器系数
def coefficient(x, y, w):
    assert len(x) == len(y)

    error_num = (x == y)

    error = np.sum([
        w_i * bool2num(error_i) for w_i, error_i in zip(w, error_num)
    ])

    return 0.5 * np.log((1 - error) / error)

# 均方差
def R2_score(y_pred, y_true): 
    return np.sum(np.square(y_pred - y_true)) / np.sum(np.square(np.mean(y_pred) - y_true))

# 更新权重因子
# @pysnooper.snoop('file.log')
def update_weight(w, alpha, y_true, y_pred):

    assert len(w) == len(y_pred) == len(y_true)

    w_new = [ w_i * np.exp(-alpha * y_i * G_i) for w_i, y_i, G_i in zip(w, y_true, y_pred)]

    total = np.sum(w_new)
    return np.array(w_new) / total

# 选择最优分类器
def optimize_errror_rate(x, y_true, w):
    '''

    :param x: x 训练集文本
    :param y_true: label值
    :return: 返回最低错误率
    '''
    # 找出label中前后不一样的标签值
    index = [
        i + 0.5  for i in range(len(y_true) - 1) if y_true[i] != y_true[i + 1]
    ]

    y_pred = [
       [sign(x_i, i) for x_i in x] for i in index
    ]
    y_pred_reverse = [
       [sign_reverse(x_i, i) for x_i in x] for i in index
    ]

    error = [errror_rate(y_pred=y_pred_i, y_true=y_true, w=w) for y_pred_i in y_pred]
    error_reverse = [errror_rate(y_pred=y_pred_i, y_true=y_true, w=w) for y_pred_i in y_pred_reverse]

    min_error = min(min(error_reverse), min(error))

    if min_error in error:

        y_pred_min_error = y_pred[error.index(min_error)]
    elif min_error in error_reverse:
        y_pred_min_error = y_pred[error_reverse.index(min_error)]

    return y_pred_min_error, min_error,
    # print(index)



if __name__ == '__main__':


    X = np.arange(10)
    Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    # 初始化weight
    w = np.ones(len(X)) / len(X)

    # 计算最优分类方式,并且输出分类错误率

    error = 1
    # ###################################################
    # 开始建立自动循环结构
    # ###################################################
    T_iter = 0
    alpha_total = []
    y_pred_total = []
    m = 10
    F_x = 0
    #
    w =np.array([ 0.04545 , 0.04545 ,0.04545,  0.16667 , 0.16667  ,0.16667,  0.10606,  0.10606,
  0.10606,  0.04545])
    y_pred, error = optimize_errror_rate(X, Y, w)

    Behold().show('y_pred', 'error')

    while m != 0 or T_iter < 6:

        y_pred, error = optimize_errror_rate(X, Y, w)
        alpha = 0.5 * np.log((1 - error) / error)
        w = update_weight(w, alpha, Y, y_pred)

        F_x += np.multiply(alpha, y_pred)

        Y_pred = [sign_reverse(i) for i in F_x]

        m = (Y_pred == Y).tolist().count(False)
        Behold().show('w', 'alpha', 'y_pred', 'Y_pred', 'm')
        T_iter += 1
        print('第{}个分类器中,分类个数为{}, 不满足条件,继续迭代'.format(T_iter, m))


#                                                   alpha_total = [alpha_1, alpha_2, alpha_3]
# y_pred_total = [y_pred_1, y_pred_2, y_pred_3]
# # G = alpha_1 * y_pred_1 + alpha_2 * y_pred_2 + alpha_3 * y_pred_3
# G = [
#     alpha_1 * j_1 + alpha_2 * j_2 + alpha_3 * j_3 for j_1, j_2, j_3 in zip(y_pred_1, y_pred_2, y_pred_3)
# ]
# G_total = [
#     sign_reverse(i) for i in G
# ]
# print(G_total == Y)
# print(G_total)
# print(G)