'''
假设有一链表,
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-
import pysnooper
class Node(object):
    def __init__(self,val,p=0):
        self.data = val
        self.next = p

class LinkList:
    def __init__(self):
        self.head = 0

    def __getitem__(self, key):

        if self.is_empty():
            print('linklist is empty.')
            return 1

        elif key <0  or key > self.getlength():
            print('the given key is error')
            return

        else:
            return self.getitem(key)

    def __setitem__(self, key, value):

        if self.is_empty():
            print('linklist is empty.')
            return

        elif key <0  or key > self.getlength():
            print('the given key is error')
            return

        else:
            self.delete(key)
            return self.insert(key)

    def initlist(self,data):

        self.head = Node(data[0])

        p = self.head

        for i in data[1:]:
            node = Node(i)
            p.next = node
            p = p.next

    def getlength(self):

        p =  self.head
        length = 0
        while p!=0:
            length+=1
            p = p.next

        return length

    def is_empty(self): return self.getlength() == 0

    def clear(self):

        self.head = 0

    def append(self,item):

        q = Node(item)
        if self.head ==0:
            self.head = q
        else:
            p = self.head
            while p.next!=0:
                p = p.next
            p.next = q

    def getitem(self,index):

        if self.is_empty():
            print('Linklist is empty.')
            return
        j = 0
        p = self.head

        while p.next!=0 and j <index:
            p = p.next
            j+=1

        if j ==index:
            return p.data

        else:

            print('target is not exist!')

    def insert(self,index,item):

        if self.is_empty() or index<0 or index >self.getlength():
            print('Linklist is empty.')
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            q = Node(item,p)
            post.next = q
            q.next = p

    def delete(self,index):

        if self.is_empty() or index<0 or index >self.getlength():
            print('Linklist is empty.')
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            post.next = p.next
    # @pysnooper.snoop('file.log')
    def index(self,value):

        if self.is_empty():
            print('Linklist is empty.')
            return

        p = self.head
        i = 0
        while p.next!=value and not p.data ==value:
            p = p.next
            i+=1

        if p.data == value:
            return i
        else:
            return -1

import time
# 链表
l = LinkList()
l.initlist([16, 23, 37, 44, 58])
start = time.clock()
for i in range(100):

    # print(l.index(58))
    # l.index(58)
    l.append(20)
end = time.clock()
t=end-start

print("linked Runtime is ：%.4f"%t)
# 顺序表

list = [16, 23, 37, 44, 58]
start = time.clock()
for i in range(100):

    # list.index(58)
    l.append(20)
    # count = 0
    # for j in list:
    #     if j == 58:
    #         index = count
    #     count += 1
end = time.clock()
t=end-start

print("list Runtime is ：%.4f"%t)
'''
当进行索引操作时, 重复10000次,结果如下:
linked Runtime is ： 0.1940732
list Runtime is ： 0.021360699999999982

当进行索引操作时, 重复100次,结果如下:
linked Runtime is ：0.0007
list Runtime is ：0.0017
'''
length = l.getlength()
# print(length)
# print(l.getitem(0))
# print(l.getitem(4))
# l.append(6)
# print(l.getitem(5))
#
# l.insert(4,40)
# print(l.getitem(3))
# print(l.getitem(4))
# print(l.getitem(5))

# l.delete(5)
# print(l.getitem(5))
#
# l.index(5)

#先定一个node的类
class Node():                  #value + next
    def __init__ (self, value = None, next = None):
        self._value = value
        self._next = next

    def getValue(self):
        return self._value

    def getNext(self):
        return self._next

    def setValue(self,new_value):
        self._value = new_value

    def setNext(self,new_next):
        self._next = new_next

#实现Linked List及其各类操作方法
class LinkedList():
    def __init__(self):      #初始化链表为空表
        self._head = Node()
        self._tail = None
        self._length = 0

    #检测是否为空
    def isEmpty(self):
        return self._head == None

    #add在链表前端添加元素:O(1)
    def add(self,value):
        newnode = Node(value,None)    #create一个node（为了插进一个链表）
        newnode.setNext(self._head)
        self._head = newnode

    #append在链表尾部添加元素:O(n)
    def append(self,value):
        newnode = Node(value)
        if self.isEmpty():
            self._head = newnode   #若为空表，将添加的元素设为第一个元素
        else:
            current = self._head
            while current.getNext() != None:
                current = current.getNext()   #遍历链表
            current.setNext(newnode)   #此时current为链表最后的元素

    import pysnooper
    @pysnooper.snoop('file.log')
    #search检索元素是否在链表中
    def search(self,value):
        current=self._head
        foundvalue = False
        while current != None and not foundvalue:
            if current.getValue() == value:
                foundvalue = True
            else:
                current=current.getNext()
        return foundvalue

    #index索引元素在链表中的位置
    def index(self,value):
        current = self._head
        count = 0
        found = None
        while current != None and not found:
            count += 1
            if current.getValue()==value:
                found = True
            else:
                current=current.getNext()
        if found:
            return count
        else:
            raise ValueError ('%s is not in linkedlist'%value)

    #remove删除链表中的某项元素
    def remove(self,value):
        current = self._head
        pre = None
        while current!=None:
            if current.getValue() == value:
                if not pre:
                    self._head = current.getNext()
                else:
                    pre.setNext(current.getNext())
                break
            else:
                pre = current
                current = current.getNext()

    #insert链表中插入元素
    def insert(self,pos,value):
        if pos <= 1:
            self.add(value)
        elif pos > self.size():
            self.append(value)
        else:
            temp = Node(value)
            count = 1
            pre = None
            current = self._head
            while count < pos:
                count += 1
                pre = current
                current = current.getNext()
            pre.setNext(temp)
            temp.setNext(current)

# link = LinkedList()


