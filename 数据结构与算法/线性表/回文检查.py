from Deque import *

def Palindrome_check(str):

    deque = Deque()
    for i in str:

        deque.addFront(i)

    return deque.removeRear() == deque.removeFront()

a = "lsdkjfskf"
b = "radar"

print(Palindrome_check(a))
print(Palindrome_check(b))