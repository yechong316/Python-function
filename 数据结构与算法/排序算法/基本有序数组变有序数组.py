'''
I.给一个长度为n的数组•共顺序本來已经排好但后来因为各种原因变为只畑jfe本刘崖的-
但其混乱度有上限k,表现为：对元素 >.设共在宪全有序后的数组中index为i』，在给与
的站本有序的数组中index为i_n.那么|i_n - i_o| <= k
要求：给与基本有序数组arr号®乱度k「将Jr变为丸全有序的.井估计时空
'''

def order_list(arr, k):

    '''
    以2K为进行堆排序，滑动步长为k，进行遍历
    :param arr:
    :param k:
    :return:
    '''
    if not arr or arr == []: return
    elif len(arr) == 1 or k <= 0: return arr
    elif k >= len(arr):
        print('{}超过数组的长度，请重新输入！'.format(k))
    else:

        n = len(arr) // k
        for i in range(n):

            arr[i * k:2*k + i * k] = heap_sort(arr[i * k:2*k + i * k])

        # arr[n * k:n] = heap_sort(arr[n * k:n])

    return arr




def heapfipy(arr, length, i):

    largest = i

    l = 2 * i + 1
    r = 2 * i + 2

    if l < length and arr[largest] < arr[l]:

        largest = l

    if r < length and arr[largest] < arr[r]:

        largest = r

    if largest != i:

        arr[i], arr[largest] = arr[largest], arr[i]

        heapfipy(arr, length, largest)

def heap_sort(arr):

    if not arr or arr == []: return
    elif len(arr) == 1: return arr
    else:

        length = len(arr)
        for i in range(length, -1, -1):

            heapfipy(arr, length, i)

        for i in range(length - 1, 0, -1):

            arr[i], arr[0] = arr[0], arr[i]

            heapfipy(arr, i, 0)
        return arr

if __name__ == '__main__':


    arr = [1, 5, 3, 4, 2, 6, 7, 8, 9, 10]

    print(arr)
    arr1 = order_list(arr, 7)
    # arr1 = heap_sort(arr)
    print('*'*50)
    print(arr1)



