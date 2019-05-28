import numpy as np


def merge_sorted(list):


    n = len(list)
    if n <= 1 :
        return list
    mid_value = n // 2
    # ####################################################
    # 分割列表
    # ####################################################

    left_li = merge_sorted(list[:mid_value])
    right_li = merge_sorted(list[mid_value:])

    left_pointer, right_pointer = 0, 0

    result = []

    while left_pointer < len(left_li) and right_pointer < len(right_li):

        if left_li[left_pointer] < right_li[right_pointer]:

            result.append(left_li[left_pointer])
            left_pointer += 1

        else:

            result.append(right_li[right_pointer])
            right_pointer += 1
    result += left_li[left_pointer:]
    result += right_li[right_pointer:]

    return result

if __name__ == '__main__':
    n_samples = 2
    list = [1, 3, -3, 7]
    print('排序前：', list)
    li = merge_sorted(list)

    print('排序后：', li)