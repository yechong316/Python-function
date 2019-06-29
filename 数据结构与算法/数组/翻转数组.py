def num_reverse(array):

    if array is None or array == []: return None

    elif len(array) == 1 or array[0] < array[len(array) - 1] or len(array) \
            == 2 and array[0] == array[1]: return 0

    else:

        index_min, index_max = 0, len(array) - 1
        while index_min != index_max:

            index_mid = (index_min + index_max) // 2

            if array[index_min] < array[index_mid]:

                index_min = index_mid
            elif array[index_min] > array[index_mid]:
                index_max = index_mid
            else:

                while array[index_min] == array[index_mid]:

                    index_min += 1
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