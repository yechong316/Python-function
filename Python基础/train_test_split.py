
def train_test_split(X, Y=None, train_size=0.2):

    '''
    将X,Y数据集乱序，并且返回，要求输入训练集的维度，默认label的维度为1

    :param X:
    :param Y:
    :param dimention: 训练集的维度
    :return: 乱序后的训练集和label
    '''

    import random
    random.shuffle(X)
    random.shuffle(Y)
    split = int(len(X) * train_size)

    x_train, x_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]
    return x_train, x_test, y_train, y_test
