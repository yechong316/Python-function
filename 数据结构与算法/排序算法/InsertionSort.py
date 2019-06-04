def Insertion_Sort(alist):
    assert alist != None, 'None type dont support sorted！'

    new_list = [alist[0]]
    for old in range(1, len(alist)):

        old_num = alist[old]
        for new in range(len(new_list) - 1, -1, -1):

            if old_num >= new_list[new]:

                # 插入排序中，索引号是多少，则插入完毕新元素的索引号即在哪里
                new_list.insert(new + 1, old_num)

                # 如果待排数字比第一个数字大，则插入到索引号1的位置，不执行下面的if条件
                break

            # 当对新序列遍历完毕，发现新数字是最小的，那么就插入到新序列的首位
            if new == 0:
                    new_list.insert(0, old_num)


    return new_list

# 二分查找法搜索
def half_interval_search_Insertion_Sort_(alist):
    assert alist != None, 'None type dont support sorted！'

    new_list = [alist[0]]


    for old in range(1, len(alist)):

        N_min, N_max = 0, len(new_list) - 1
        num = alist[old]
        split = (N_max + N_min) // 2

        while ((N_max - N_min) // 2) != 0:

            if num <= new_list[split - 1]:

                N_max = split
            else:

                N_min = split

            split = (N_max + N_min) // 2

        if N_min == N_max:

            if num <= new_list[N_max]: new_list.insert(N_max, num)

            else: new_list.insert(N_max + 1, num)
        else:

            if num <= new_list[N_min]: new_list.insert(N_min, num)


            elif num <= new_list[N_max]: new_list.insert(N_max, num)

            elif num > new_list[N_max]: new_list.insert(N_max + 1, num)
    return new_list

if __name__ == '__main__':
    list = [30, -120, 65, 69, 78, 80]
    l = half_interval_search_Insertion_Sort_(list)
    print(l)