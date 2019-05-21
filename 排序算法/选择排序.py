def Selection_Sort(alist):
    assert alist != None, 'Ntype dont support sorted！'

    new = []
    while len(alist) != 1:

        small_num = alist[0]
        for j in range(1, len(alist)):

            if small_num > alist[j]:
                small_num = alist[j]

        alist.remove(small_num)
        new.append(small_num)
    new.append(alist[0])
    return new


if __name__ == '__main__':
    list = [-30, 20,-10,50,250]
    l = Selection_Sort(list)
    print(l)