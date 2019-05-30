class Queue:

    def __init__(self): self._list = []

    def add_head(self, ele):  self._list.insert(0, ele)

    def add_end(self, ele):  self._list.insert(len(self._list), ele)
    

    def pop(self):

        if self.is_empty():
            return
        a = self._list.pop(-1)
        return a

    def peek(self): return self._list[-1]

    def is_empty(self): return len(self._list) == 0

    def size(self): return len(self._list)

if __name__ == '__main__':

    s = Queue()
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

