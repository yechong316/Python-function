import numpy as np

def mess_dataset_order(train_data, train_label=None, dimention=2):

    '''
    将X,Y数据集乱序，并且返回，要求输入训练集的维度，默认label的维度为1

    :param train_data:
    :param train_label:
    :param dimention: 训练集的维度
    :return: 乱序后的训练集和label
    '''
    # assert dimention in [1, 2, 3, 4], '维度必须在1~4之间，否则请修改函数'
    assert type(dimention) == int

    permutation = np.random.permutation(train_label.shape[0])

    shuffled_labels = train_label[permutation]
    shuffled_dataset = train_data[permutation, :]
    if dimention == 3: shuffled_dataset = train_data[permutation, :, :]
    elif dimention == 4: shuffled_dataset = train_data[permutation, :, :, :]


    return shuffled_dataset, shuffled_labels

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