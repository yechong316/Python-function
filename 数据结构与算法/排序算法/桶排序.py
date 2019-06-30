#!/usr/bin/python
# -*- coding: utf-8 -*-
# 桶排序

def bucket_sort(the_list):
    # 设置全为0的数组
    all_list = [0 for i in range(100)]
    last_list = []
    for v in the_list:
        all_list[v] = 1 if all_list[v] == 0 else all_list[v] + 1
    for i, t_v in enumerate(all_list):
        if t_v != 0:
            for j in range(t_v):
                last_list.append(i)
    return last_list


if __name__ == '__main__':
    the_list = [10, 1, 18, 30, 23, 12, 7, 5, 18, 17]
    # print
    # "排序前：" + str(the_list)
    # print
    # "排序后：" + str(bucket_sort(the_list))