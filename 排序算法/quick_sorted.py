import numpy as np
def qucik_sorted(alist):

    n = len(alist)
    low = 0
    high = n -1
    mid_value = alist[0]

    while low < high:
        while low < high and alist[low] < mid_value:

            alist[]
    pass

if __name__ == '__main__':
    n_samples = 9
    alist = np.random.randint(-20, 20, size=n_samples)
    print('排序前：', alist)
    l = qucik_sorted(alist)

    print('排序后：', alist)