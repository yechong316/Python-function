import numpy as np


def quick_sorted(list, first, last):

    coutn = 1
    if first >= last:
        coutn += 1
        # print('仅有一个元素，无需排序！',coutn)
        return

    mid_value = list[first]

    low = first
    high = last

    while low < high:

        while low < high and list[high] >= mid_value:
            high -= 1
        list[low] = list[high]

        while low < high and list[low] < mid_value:
            low += 1
        list[high] = list[low]


    list[low] = mid_value
#   对左边排序
    quick_sorted(list, first, low - 1)
    quick_sorted(list, low + 1, last)

    return list

if __name__ == '__main__':
    n_samples = 9
    alist = np.random.randint(-20, 20, size=n_samples)
    print('排序前：', alist)
    l = quick_sorted(alist, 0, len(alist) - 1)

    print('排序后：', l)