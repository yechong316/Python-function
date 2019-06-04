import numpy as np
def shell_sorted(alist):

    '''
    1. 初始化参数定义，间隔
    2.
    '''
    length = len(alist)
    increment = length // 2

    while increment != 0:
        indexs = []
        i = 0
        while i < increment:

            k = 0
            child_indexs = []
            indexs.append(child_indexs)
            while i + increment * k <= length - 1:

                child_indexs.append(i + increment * k)
                k += 1

            i += 1
        # 已提取到本次各个列表的索引号
        new = [1] * length
        for index in indexs:

            length_child = len(index)
            for i in range(2, length_child + 1):

                for j in range(0, length_child - i + 1):

                    if alist[index[j]] > alist[index[j + 1]]:
                        alist[index[j]], alist[index[j + 1]] = alist[index[j + 1]], alist[index[j]]

        increment -= 1
    return alist


if __name__ == '__main__':
    n_samples = 9
    alist = np.random.randint(-30, 30, size=n_samples)
    print('排序前：', alist)
    l = shell_sorted(alist)

    # l = [1] * 10
    print('排序后：', l)