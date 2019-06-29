def num_reverse(array):

    # 数组为空或[]直接返回
    if array is None or array == []: return None

    # 当数组长度为1，首数字小于末尾数字，长度为2且首尾数字相等时返回0
    elif len(array) == 1 or array[0] < array[len(array) - 1] or len(array) \
            == 2 and array[0] == array[1]: return 0

    else:

        # 创建两个指针
        index_min, index_max = 0, len(array) - 1

        # 当两个指针重叠时，返回当前小索引
        while index_min != index_max:

            index_mid = (index_min + index_max) // 2

            # 证明临界点在右边，min索引往右跑
            if array[index_min] < array[index_mid]:

                index_min = index_mid

            # 证明临界点在左边，max索引往左跑
            elif array[index_min] > array[index_mid]:
                index_max = index_mid
            else:

                # 当相等时，只能采用顺序遍历来查找
                while array[index_min] == array[index_mid]:

                    # 游标不断往右跑，找到可以时两个游标不相等的值
                    index_min += 1

                    # 当中间游标位移翻转部分数组时，直接break
                    if index_min == index_mid:

                        break
        return index_min

if __name__ == '__main__':

    a = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,123,10,10,10,10,10,10]
    # a = [3, 4, 1, 2]
    # a = [3, 4, 1, 2]
    # a = [3, 4, 1, 2]
    num = num_reverse(a)
    print('最小翻转次数', num)