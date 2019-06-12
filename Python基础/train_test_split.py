import numpy as np

def train_test_split(X, Y=None, train_size=0.2):

    '''
    将X,Y数据集乱序，并且返回，要求输入训练集的维度，默认label的维度为1

    :param X:
    :param Y:
    :param dimention: 训练集的维度
    :return: 乱序后的训练集和label
    '''

    if type(X) != np.ndarray:
        X = np.array(X)
    x = X[np.random.permutation(len(X))]
    split = int(len(x) * train_size)

    x_train, x_test = x[:split], x[split:]

    if Y:
        y = Y[np.random.permutation(len(Y))]
        y_train, y_test = y[:split], y[split:]
        return x_train, x_test, y_train, y_test
    return x_train, x_test
if __name__ == '__main__':
    train_data = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])

    train_label = np.array([
        [1.5],
        [3.5],
        [4.5],
        [5.5],
    ])

    # shuffled_dataset, shuffled_labels = mess_dataset_order(train_data, train_label, dimention=2)
    print()