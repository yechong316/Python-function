def heapify_small(arr, n, i):

    largest = i
    l = i * 2 + 1
    r = i * 2 + 2

    if l < n and arr[largest] < arr[l]:

        largest = l
    if r < n and arr[largest] < arr[r]:

        largest = r

    if i != largest:

        arr[i], arr[largest] = arr[largest], arr[i]

        heapify_small(arr, n, largest)

def heap_sort(arr):

    n = len(arr)
    for i in range(n - 1, -1, -1):

        heapify_small(arr, n, i)

    for i in range(n - 1, 0, -1):

        arr[i], arr[0] = arr[0], arr[i]
        heapify_small(arr, i, 0)

    return arr



def find_min_k(arr, k):

    if arr is None or k <= 0:

        return
    elif k >= len(arr): return arr
    else:

        result = arr[:k]
        result = heap_sort(result)
        n = len(arr)
        for i in range(k, n):

            if arr[i] < result[-1]:

                result[-1] = arr[i]
                result = heap_sort(result)
        return result

def find_max_k(arr, k):

    if arr is None or k <= 0:

        return
    elif k >= len(arr): return arr
    else:

        result = arr[:k]
        result = heap_sort(result)
        n = len(arr)
        for i in range(k, n):

            if arr[i] > result[0]:

                result[-1] = arr[i]
                result = heap_sort(result)
        return result

arr = [90, 50, 80, 16, 30, 60, 70, 10, 2]
# heap_sort(arr)
# n = len(arr)
# print("排序后")
# print(arr)
# result = find_min_k(arr, 10)
result = find_max_k(arr, 2)
print(result)
