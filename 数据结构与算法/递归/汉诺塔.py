class Stack:

    def __init__(self):

        self.l = []

    def push(self, ele):

        self.l.append(ele)

    def pop(self):

        num = self.l[-1]
        del self.l[-1]
        return num


def HanoiTower(num, a=None, b=None, c=None):

    if num == 0: return 0

    a = Stack()
    b = Stack()
    c = Stack()
    for i in range(num):
        a.push(i)

    num1 = HanoiTower(num - 1, a, c, b)
    c.push(a.pop())

    num2 = HanoiTower(num - 1, b, a, c)

    return num1 + num2 + 1



if __name__ == '__main__':

    num = 3
    print("移动次数： ", HanoiTower(num))
    pass
