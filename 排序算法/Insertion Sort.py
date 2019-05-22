def Insertion_Sort(alist):
    assert alist != None, 'None type dont support sorted！'

    new_list = [alist[0]]
    for old in range(1, len(alist)):

        old_num = alist[old]
        for new in range(len(new_list) - 1, -1, -1):

            if old_num >= new_list[new]:

                new_list.insert(new + 1, old_num) # 插入排序中，索引号是多少，则插入完毕新元素的索引号即在哪里
                break

            if new == 0:
                    new_list.insert(0, old_num)
                # break
        # alist.remove(old_num)


    return new_list


if __name__ == '__main__':
    list = [-30,10,-50, -350 ]
    l = Insertion_Sort(list)
    print(l)