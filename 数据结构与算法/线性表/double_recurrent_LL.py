class Node:

    def __init__(self, ele):
        assert ele != None, '创建节点时，不允许传入空值，请重新输入！'

        # 双向链表里面的节点包含3部分内容，数据本身+前指针+后指针
        self.ele = ele
        self.next = None
        self.pre = None


class Double_link_list:

    def __init__(self, node=None):

        self._head = node

    def add(self, item):

        # 当链表为空时，初次添加节点使用此函数，定义二者之间的关系
        node = Node(item)
        self._head = node
        node.pre = self._head
        node.next = self._head

    def append(self, item):

        # 给链表末尾添加元素时
        node = Node(item)

        # 判断是否为空，空链表之间采用上面的add函数
        if self.is_empty():
            self.add(item)

        first_node = self._head
        # 如果当前节点的下一个节点指向空，即遍历至最后一个节点时，终止遍历
        current_node = self._head
        count = 1
        while current_node.next != first_node:
            current_node = current_node.next
            count += 1

        # 定义二者之间的关系
        current_node.next = node
        node.pre = current_node
        node.next = first_node

    def travel(self):

        # 功能：依次打印出链表的元素
        cur = self._head
        if self.is_empty():
            print('当前链表为空！')
        else:
            first_node = self._head
            # 当节点指向None时，终止运行
            while cur.next != first_node:
                print(cur.ele, end=' ')
                cur = cur.next
            print(cur.ele)

    def insert(self, pos, ele):

        # 在链表中间位置添加元素，要考虑3点：跟pos小于0或者大于链表长度的特殊情况需要特殊对待
        if pos < 0 :

            node = Node(ele)

            # 定义带插节点与当前节点的关系
            first_node = self._head
            node.next = first_node
            first_node.pre = node

            # 定义dai插节点与当前节点的前一个节点的关系
            self._head = node
            node.pre = self._head


        elif pos >= self.length() - 1: self.append(ele)

        else:

            current_node = self._head
            node = Node(ele)

            count = 1
            while count <= pos:

                current_node = current_node.next
                count += 1

            # 构建插入节点之后的节点与待插入节点之间的关系
            final_node = current_node.next
            node.next = final_node
            final_node.pred = node

            # 构建插入节点之前的节点与待插入节点之间的关系
            current_node.next = node
            node.pre = current_node

    def is_empty(self): return self._head == None

    def length(self):

        if self.is_empty(): return 0

        first_node = self._head
        # 计数器
        count = 1
        current_node = self._head

        # current_node = current_node.next
        while current_node.next != first_node:
            count += 1
            current_node = current_node.next
        # print(count)
        return count

    def delete_node(self, ele):
        '''
        根据元素名称来删除节点
        :param ele: 待删除节点的元素
        :return: 无
        '''
        cur = self._head
        if self.is_empty() == None: print('当前链表为空')

        count = 1
        while count <= self.length(): # 计数器从1开始直至整个链表

            # 如果当前元素是要被删除的元素进入下面的判断条件里面，不是则遍历至下一个节点
            if cur.ele == ele:

                back_node = cur.next

                if count == 1: # 被删除节点是第一个，则仅仅需要将头指针指向第二个节点即可

                    self._head = back_node
                elif count == self.length():# 被删除节点是最后一个，则倒数第二个节点的next指向空
                    pre_node = cur.pre
                    pre_node.next = None
                else:

                    # 被指向节点是中间节点，则进入被删除节点的前后分别定义指针关系即可
                    pre_node = cur.pre
                    pre_node.next = back_node
                    back_node.pre = pre_node
            count += 1
            cur = cur.next

    def search(self, ele):

        cur = self._head
        if self.is_empty() == None: print('当前链表为空')

        count = 1
        while count <= self.length():  # 计数器从1开始直至整个链表

            # 如果当前元素是要被删除的元素进入下面的判断条件里面，不是则遍历至下一个节点
            if cur.ele == ele:

                print('节点位置：{}'.format(count))
                # 跳出循环体，默认只搜索第一个发现的元素
                break

            count += 1
            cur = cur.next

        if count > self.length():
            print('未找到该节点')

    def list2ll(self, list):

        for i in list:

            self.append(i)

    def final_node(self):

        if self.is_empty(): return

        count = 1
        cur = self._head
        while count != self.length():

            cur = cur.next
            count += 1

        return cur

if __name__ == '__main__':

    # a = Node(2)
    ll = Double_link_list()
    # ll.travel()
    ll.add('a')
    # # ll.travel()
    ll.append('f')
    ll.append('c')
    ll.append('e')
    # ll.append('f')
    ll.append('g')
    # ll.append('h')
    ll.travel()
    length = ll.length()
    final_node = ll.final_node()
    print('链表长度为：', length)
    print('final_node：', final_node.ele)
    # ll.insert(-10, 'd')
    # ll.travel()
    # ll.delete_node('a')
    # ll.travel()
    # ll.search('f得到')
    # print(ll.is_empty())
    # print(ll.length())