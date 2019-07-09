from single_linked_list import *

class Stack:

    def __init__(self):

        self.stack = []

        pass

    def push(self, ele):
        self.stack.append(ele)

    def pop(self):

        data = self.stack[-1]
        del self.stack[-1]
        return data

    def is_empty(self):
        return self.stack == []



def reverse_list(head):

    if not head: return None
    else:

        cur = head
        s = Stack()
        while cur:
            s.push(cur.ele)
            cur = cur.next

        while not s.is_empty():

            a = s.pop()
            print(a)



if __name__ == '__main__':

    l = [1, 2, 3, 4, 5]
    ll = Link_list()
    for i in l:

        ll.append(i)
    reverse_list(ll._head)