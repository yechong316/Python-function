
class MinHeap():
    def parent(self,n):
        return (n-1)//2
    def leftChild(self,n):
        return 2*n+1
    def rightChild(self,n):
        return 2*n+2

    #将list的前n个值构建为最小堆
    def build_min_heap(self,n,list):
        for i in range(n):
            t=i
            while (t!=0 and (list[self.parent(t)] > list[t])):
                temp = list[t]
                list[t] = list[self.parent(t)]
                list[self.parent(t)] = temp
                t = self.parent(t)
    #将数据与堆顶进行比较
    def adjust_heap(self,i,n,list):
        if(list[i]<list[0]):
            return
        #置换堆顶
        temp = list[i]
        list[i] = list[0]
        list[0] = temp
        #调整最小堆
        t=0
        while(self.leftChild(t)<n and list[t]>list[self.leftChild(t)]) or (self.leftChild(t)<n and list[t]>list[self.leftChild(t)]):
            if ((self.rightChild(t) < n )and (list[self.rightChild(t)] < list[self.leftChild(t)])):
                #右孩子更小，置换右孩子
                temp = list[t]
                list[t] = list[self.rightChild(t)]
                list[self.rightChild(t)] = temp
                t = self.rightChild(t)
            else:
                #否则置换左孩子
                temp = list[t]
                list[t] = list[self.leftChild(t)]
                list[self.leftChild(t)] = temp
                t = self.leftChild(t)
    def findTopN(self,n,list):
        self.build_min_heap(n,list)
        for i in range(len(list)):
            self.adjust_heap(i,n,list)
    def print(self,list):
        print(list)

if __name__ == "__main__":
    n_samples = 50
    from 桶排序 import bucket_sort
    import numpy as np
    list = np.random.randint(-20, 20, size=n_samples)
    print("原数组：")
    from datetime import datetime
    print('堆泡排序耗时：', list)
    minHeap = MinHeap()
    # minHeap.print(list)
    start = datetime.now()
    repeat = 100
    # for _ in range(repeat):

    minHeap.findTopN(1, list)
    end = datetime.now()
    print('堆泡排序耗时：', list)
    
    
    # print("调整后数组：")
    # minHeap.print(list)
    # start = datetime.now()
    # for _ in range(repeat):
    #
    #     quick_sorted(list, 1, len(list)-1)
    # end = datetime.now()
    # print('快速排序耗时：', end-start)
    #
    # start = datetime.now()
    # for _ in range(repeat):
    #
    #     bucket_sort(list)
    # end = datetime.now()
    # print('桶泡排序耗时：', end-start)
    #
    # # minHeap.print(list)
    # start = datetime.now()
    # for _ in range(repeat):
    #
    #     bubble_sort(list)
    # end = datetime.now()
    # print('冒泡排序耗时：', end-start)
