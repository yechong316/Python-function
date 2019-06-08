
def quick_sorted(list, first, last):

    coutn = 1
    if first >= last:
        coutn += 1
        print('仅有一个元素，无需排序！',coutn)
        return

    mid_value = list[first]

    low = first
    high = last

    while low < high:

        while low < high and list[high] >= mid_value:
            high -= 1
        list[low] = list[high]

        while low < high and list[low] < mid_value:
            low += 1
        list[high] = list[low]


    list[low] = mid_value
#   对左边排序
    quick_sorted(list, first, low - 1)
    quick_sorted(list, low + 1, last)

    return list


class Solution:
    def removeDuplicates(self, aList):

        self.unique_list = []

        for i in aList:

            if i not in self.unique_list:
                self.unique_list.append(i)

        return quick_sorted(self.unique_list,0, len(self.unique_list) - 1)


if __name__ == '__main__':
    a = [1, 1,1,1,1,1, 3, 44, 3,4, 5, 5]

    b = Solution()
    b.removeDuplicates(a)
    print(b.unique_list)