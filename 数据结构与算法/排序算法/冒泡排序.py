def bubble_sort(alist):

    assert alist != None, 'Ntype dont support sorted！'

    length = len(alist)
    for i in range(1, length):

        
        for j in range(0, length - i):

            if alist[j] > alist[j + 1]:
                alist[j], alist[j + 1] = alist[j + 1], alist[j]

    return alist

if __name__ == '__main__':

    list = [-30, -1000, 50, -89]
    l = bubble_sort(list)
    print(l)