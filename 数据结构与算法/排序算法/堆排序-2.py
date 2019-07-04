def heapify_large(arr, n, i):
    largest = i
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    if l < n and arr[i] > arr[l]:
        largest = l

    if r < n and arr[largest] > arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换

        heapify_large(arr, n, largest)
def heapify_small(arr, n, i):
    largest = i
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    if l < n and arr[i] > arr[l]:
        largest = l

    if r < n and arr[largest] > arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换

        heapify_small(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Build a maxheap. 
    for i in range(n, -1, -1):
        heapify_small(arr, n, i)

        # 一个 b个交换元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换
        heapify_small(arr, i, 0)


arr = [90, 50, 80, 16, 30, 60, 70, 10, 2]
heap_sort(arr)
n = len(arr)
print("排序后")
print(arr)
# for i in range(n):
#     print("%d" % arr[i]),