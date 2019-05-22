import InsertionSort as px1
import 冒泡排序 as px2
import 选择排序 as px3
import time
# , 选择排序, InsertionSort
# bubble

if __name__ == '__main__':

    list = [100, 99, 50 , 12 , 9, -10, -80]
    number = 100000

    f = [px1.Insertion_Sort, px1.half_interval_search_Insertion_Sort_, px2.bubble_sort, px3.Selection_Sort]
    name = ['插入排序', '二分搜索插入排序', '冒泡排序', '选择排序']

    for j in range(len(f)):

        begin =  time.time()
        for i in range(number):
            f[j](list)
        end =  time.time()
        cost_time = end - begin
        print('运行{}次后，{}消耗时间：{}'.format(number, name[j], cost_time))