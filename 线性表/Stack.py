class Stack:

    def __init__(self): self._list = []

    def push(self, ele): self._list.append(ele)

    def pop(self):

        if self.is_empty():
            return
        a = self._list.pop(-1)
        return a


    def peek(self): return self._list[-1]

    def is_empty(self): return len(self._list) == 0

    def size(self): return len(self._list)

if __name__ == '__main__':

    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    s.push(8)
    a = s.peek()
    # print(a)
    # print(s.pop())
    print(s.size())
    # print(s.pop())
    # print(s.pop())
    # print(s.pop())
    # print(s.pop())

