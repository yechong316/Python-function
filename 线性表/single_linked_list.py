class Node:

    def __init__(self, ele):
        assert ele != None, '创建节点时，不允许传入空值，请重新输入！'

        self.ele = ele
        self.next = None

class Link_list:

    def __init__(self, node=None):

        self._head = node

    def add(self, item):

        node = Node(item)
        self._head = node

    def append(self, item):

        node = Node(item)
        if self.is_empty():
            self.add(item)
        else:

            current_node = self._head
            while current_node.next != None:
                current_node = current_node.next

            current_node.next = node

    def travel(self):

        cur = self._head
        if self.is_empty():
            print('当前链表为空！')
        else:
            while cur != None:
                print(cur.ele, end=' ')
                cur = cur.next
        print('')

    def insert(self, pos, ele):
        if pos <= 0 :
            self.add(ele)
        elif pos > (self.length()):
            self.append(ele)
        else:
            current_node = self._head
            node = Node(ele)
            count = 0
            while count < (pos - 1):
                current_node = current_node.next
                count += 1
            node.next = current_node.next
            current_node.next = node

    def is_empty(self): return self._head == None

    def length(self):
        count = 0
        current_node = self._head

        while current_node != None:
            count += 1
            current_node = current_node.next
        # print(count)
        return count

    def list2link_list(self, list):

        for i in list:

            self.append(i)

    def search_node(self, index):
        '''
        等价于列表中的索引功能
        :param index: 列表中的索引号
        :return:
        '''
        if self.is_empty():
            print('当前链表为空！')
            return None
        else:
            # 索引号小于0， 则print第一个节点的编号
            if index < 0:

                return self._head.ele

            elif index > self.length():

                # 索引号大于程度， 则print最后一个节点的编号
                cur = self.length()
                while cur.next != None:
                    cur = cur.next

                return cur.ele
            else:

                # 索引号在中间
                current_node = self._head
                count = 0
                while count != index:
                    current_node = current_node.next
                    count += 1

                return current_node.ele

    def modefiled(self, index, new_ele):
        if self.is_empty():
            print('当前链表为空！')
            return None
        else:
            # 索引号小于0， 则替换第一个节点的编号
            if index < 0:

                self._head.ele = new_ele

            elif index > self.length():

                # 索引号大于程度， 则替换最后一个节点的编号
                cur = self.length()
                while cur.next != None:
                    cur = cur.next

                cur.ele = new_ele
            else:

                # 索引号在中间
                current_node = self._head
                count = 0
                while count != index:
                    current_node = current_node.next
                    count += 1

                current_node.ele = new_ele



if __name__ == '__main__':

    a = Node(2)
    ll = Link_list()
    # ll.travel()
    ll.add('a')
    ll.travel()
    ll.append('b')
    ll.append('c')
    ll.travel()
    ll.length()
    ll.insert(2, 'd')
    ll.travel()
    # print(ll.is_empty())
    # print(ll.length())