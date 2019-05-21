class Node:

    def __init__(self, ele):
        assert ele != None, '创建节点时，不允许传入空值，请重新输入！'

        self.ele = ele
        self.next = None
        self.pre = None

class Double_link_list:

    def __init__(self, node=None):

        self._head = node

    def add(self, item):

        node = Node(item)
        self._head = node

    def append(self, item):

        node = Node(item)
        if self.is_empty():
            self.add(node)
        current_node = self._head
        while current_node.next != None:
            current_node = current_node.next

        current_node.next = node
        node.pre = current_node
    def travel(self):

        cur = self._head
        if self.is_empty():
            print('当前链表为空！')
        else:
            while cur.next != None:
                print(cur.ele, end=' ')
                cur = cur.next
            print(cur.ele)

    def insert(self, pos, ele):
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
        count = 1
        current_node = self._head

        while current_node.next != None:
            count += 1
            current_node = current_node.next
        print(count)
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

        while cur.ele != ele:
            if cur.next == None:
                print('当前链表不存在该元素')
                break
            cur = cur.next
            count += 1

        if cur.ele == ele:

            back_node = cur.next

            pre_node = cur.pre

            if pre_node == self._head:
                self._head = back_node
            else: pre_node.next = back_node

            if back_node: back_node.pre = pre_node
        # else:

    def search(self, ele):

        count = 1
        cur = self._head
        if self.is_empty():
            print('当前链表为空，不存在该元素')
            return False

        while cur !=None:

            if cur.ele == ele:
                print('{}是第{}个节点'.format(ele, count))

                return True
            cur = cur.next
            count += 1

        # if count == self.length():
        #
        # pass







if __name__ == '__main__':

    # a = Node(2)
    ll = Double_link_list()
    # ll.travel()
    ll.add('a')
    # ll.travel()
    ll.append('b')
    ll.append('f')
    ll.append('c')
    ll.append('e')
    # ll.travel()
    ll.length()
    ll.insert(-10, 'd')
    ll.travel()
    ll.delete_node('e')
    ll.travel()
    ll.search('c')
    # print(ll.is_empty())
    # print(ll.length())