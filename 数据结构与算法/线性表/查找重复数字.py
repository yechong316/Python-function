'''
Creat by HuangDandan
2018-08-18

解题思路：
思路1-直接对数组进行排序，排序方法可以任选，然后进行顺序遍历 时间复杂度和排序方法有关
思路2-顺序扫描数组，利用哈希表检查是否出现过：时间复杂度O(n),空间复杂度O(n)
思路3-顺序扫描数组，在数组相对应的位置i上看数字是否是i: Lst[i] == i
如果是，继续遍历下一个；
如果不是，计数值为m,比较j与m位置的数字：
    如果相等，则返回重复数字
    如果不相等，则交换i和m位置对应的数值，继续循环
参考博客：https://blog.csdn.net/weixin_41427758/article/details/80152005题目描述：
'''

def mothod1(Lst):
    #冒泡排序,排序后的列表temp
    #i循环的次数
    #j两两比较的次数
    for i in range(len(Lst)-1):
        for j in range(len(Lst)-i-1):
            if Lst[j] > Lst[j+1]:
                Lst[j],Lst[j+1] = Lst[j+1], Lst[j]

    for k in range(0,len(Lst)-1):
        if Lst[k+1] == Lst[k]:
            return Lst[k]

import numpy as np
def mothod2(a):
    num_array = np.zeros(len(a))
    for i in range(len(a)):
        if num_array[a[i]] == 0:
            num_array[a[i]] += 1
        else:
            return a[i]


def mothod3(Lst):   #返回一个重复的值
    for index, value in enumerate(Lst):
        while value != index:
            if Lst[value] == value:
                return value
            else:
                Lst[index], Lst[value] = Lst[value], Lst[index]




def mothod4(a): #返回一个重复的值
    for i in range(len(a)):
        while a[i] != i:    #非常关键，当这个条件满足时，一直执行，直到不满足为止
            if a[a[i]] == a[i]:
                return a[i]
            else:
                a[a[i]], a[i] = a[i], a[a[i]]   #目标是遍历交换使得a[a[i]] == a[i]成立，返回a[i],一定可以找到，所以不会是死循环

def mothod5(a): #返回多个重复的值，时间很慢
    temp = []
    for i in range(len(a)):
        while a[i] != i:    #非常关键，当这个条件满足时，一直执行，直到不满足为止
            if a[a[i]] == a[i]:
                temp.append(a[i])
            else:
                a[a[i]], a[i] = a[i], a[a[i]]   #目标是遍历交换使得a[a[i]] == a[i]成立，返回a[i],一定可以找到，所以不会是死循环
    return temp

if __name__ == "__main__":
    Lst1 = [0,1,2,3,4,6,4]
    Lst2 = [0,1,2,3,4,6,6]
    Lst3 = [0,1,2,3,4,6,4,6]
    Lst4 = [2,5,4,2,5,3]
    Lst5 = [1, 0]
    print(Lst1)
    print("----------------------------------------")
    print(max(Lst1))
    print(mothod4(Lst1))
    print(mothod2(Lst1))
    mothod3(Lst3)
    # Lst2 = [None for i in range(10)]
    # print(Lst2)



