class Node:

    def __init__(self, ele=None):

        self.ele = ele
        self.post = None
        self.next = None


class Deque:

    def __init__(self):

        self._head = None

    def addFront(self, item):
        node = Node(item)
        if self.isEmpty():

            self._head = node
        else:
            first_node = self._head
            node.next = first_node
            self._head = node

    def addRear(self, item):

        if self.isEmpty():
            self.addFront(item)
        else:
            node = Node(item)
            cur = self._head
            while cur.next != None:

                cur = cur.next
            cur.next = node

    def removeFront(self):
        if self.isEmpty(): return
        else:

            first = self._head
            item = first.ele
            cur = first.next
            self._head = cur

            return item

    def removeRear(self):
        if self.isEmpty(): return
        elif self._head.next == None: self._head = None
        else:

            cur = self._head
            while cur.next.next != None:
                cur = cur.next

            last = cur.next
            item = last.ele
            cur.next = None
            return item

    def isEmpty(self): return self._head == None


    def size(self):

        count = 0
        if self._head == None: return count
        else:

            cur = self._head
            count += 1
            while cur.next != None:

                cur = cur.next
                count += 1
        print(count)
        return count

    def travel(self):

        cur = self._head

        while cur != None:
            print(cur.ele, end=' ')
            cur = cur.next
        print('')
if __name__ == '__main__':

    a = [3, 4, 1, 5, 9]

    deque = Deque()
    for i in a:
        deque.addRear(i)



    deque.travel()

    print(deque.removeRear())
    # deque.removeFront()
    # deque.removeFront()
    # deque.removeFront()
    # deque.removeFront()
    # deque.removeFront()
    # deque.removeFront()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.removeRear()
    # deque.travel()
    # deque.size()
    # print(deque.isEmpty())
