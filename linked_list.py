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
        node.next = self._head
        self._head = node

    def append(self, item):

        node = Node(item)
        if self.is_empty():
            self._head = node
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
        print(count)
        return count

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
    ll.insert(-10, 'd')
    ll.travel()
    # print(ll.is_empty())
    # print(ll.length())