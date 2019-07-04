def heapify_small(arr, n, i):
    
    largest = i
    l = i * 2 + 1
    r = i * 2 + 2
    
    if l < n and arr[largest] < arr[l]:
        
        largest = l
    if r < n and arr[largest] < arr[r]:
        
        largest = r
        
    if largest != i:
        
        arr[largest], arr[i] = arr[i], arr[largest]
        
        heapify_small(arr, n, largest)

def heap_sort(arr):

    n = len(arr)
    for i in range(n - 1, -1, -1):

        heapify_small(arr, n, i)

    for i in range(n - 1, 0, -1):

        arr[i], arr[0] = arr[0], arr[i]
        heapify_small(arr, i, 0)

    return arr

arr = [90, 50, 80, 16, 30, 60, 70, 10, 2]
heap_sort(arr)
n = len(arr)
print("排序后")
print(arr)
# for i in range(n):
    