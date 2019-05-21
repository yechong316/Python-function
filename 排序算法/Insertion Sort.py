def Insertion_Sort(alist):
    assert alist != None, 'Ntype dont support sorted！'

    new_list = [alist[0]]
    for old in range(1, len(alist)):

        old_num = alist[old]
        for new in range(len(new_list) - 1, 0, -1):

            if old_num >= new_list[new]:

                new_list.insert(new + 1, alist[old])
        # alist.remove(old_num)


    return new_list


if __name__ == '__main__':
    list = [-30,10,4324 ]
    l = Insertion_Sort(list)
    print(l)