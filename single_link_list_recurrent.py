from single_linked_list import Node

class RecurrentLinkList:

    def __init__(self, node=None):

        self._head = node
        if node:
            node.next = node

    def add(self, item):

        node = Node(item)
        self._head = node
        node.next = node

    def append(self, item):

        node = Node(item)
        if self.is_empty():

            self.add(node)
        else:

            first_node = self._head
            current_node = self._head
            while current_node.next != first_node:
                current_node = current_node.next

            current_node.next = node
            node.next = first_node

    def travel(self):

        cur = self._head
        if self.is_empty():
            print('当前链表为空！')


        # print(cur.ele, end=' ')
        while cur.next != self._head:
            print(cur.ele, end=' ')
            cur = cur.next
        print(cur.ele)
        # print('0')

    def insert(self, pos, ele):
        if pos <= 0 :

            node = Node(ele)
            cur = self._head
            first_node = self._head
            while cur.next != self._head:

                cur = cur.next
            cur.next = node
            self._head = node
            node.next = first_node

        elif pos > self.length():
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

        if self._head == None:
            return 0
        else:
            count = 1
            current_node = self._head
            first_node = self._head
            while current_node.next != first_node:
                count += 1
                current_node = current_node.next

            print(count)
            return count

if __name__ == '__main__':

    # a = Node(2)
    ll = RecurrentLinkList()
    # ll.travel()
    ll.add('a')
    ll.length()

    ll.append('b')
    ll.travel()
    ll.length()

    ll.append('c')
    ll.travel()
    ll.length()

    ll.insert(10, 'd')
    ll.travel()
    ll.length()
    # ll.insert(2, 'd')
    # ll.travel()