
def more_than_half_num_1(arr):


    if arr is None or arr == []: return None
    elif len(arr) == 1: return arr[0]
    else:

        dic = {}
        for i in arr:

            if i in dic:
                dic[i] += 1

            else:

                dic[i] = 1
        result = []
        for i in dic.items():

            if i[1] > len(arr) // 2:
                result.append(i[0])
        return ''.join(result)


def more_than_half_num_2(arr):

    if arr is None or arr == []: return None

    elif len(arr) == 1: return arr[0]

    else:

        # 利用快速排序的思路
        times = 0
        result = arr[0]
        for i in range(len(arr)):

            if times == 0:

                result = arr[i]
                times = 1
            elif arr[i] == result:

                times += 1
            else:

                times -= 1

        if times == 0:
            print('未找到合乎要求的数！')
        else:

            print('找到合乎要求的数！', result)



more_than_half_num_2('fabbb')
# print(result)