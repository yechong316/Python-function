# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        target = ListNode(0)  # 作为根节点的引用
        p = target
        add = 0  # 作为上一次相加是否需要进1的依据

        while l1 and l2:

            p.next = ListNode((l1.val + l2.val + add) % 10)
            add = (l1.val + l2.val + add) // 10
            p, l1, l2 = p.next, l1.next, l2.next
        l1 = l1 if l1 else l2

        while add:

            # 当某一个链表长度大于另外一个链表时，继续相加
            if l1:
                p.next = ListNode((l1.val + add) % 10)
                add = (l1.val + add) // 10
                p, l1 = p.next, l1.next
            else:
                p.next = ListNode(add)
                p = p.next
                break
        p.next = l1

        return target.next



if __name__ == '__main__':

    num1 = [2, 4, 8]
    num2 = [5, 6, 4]

    l1 = ListNode(0)
    p1 = l1
    l2 = ListNode(0)
    p2 = l2
    for i in num1:

        temp = ListNode(i)
        p1.next = temp
        p1 = temp


    for i in num2:
        temp = ListNode(i)
        p2.next = temp
        p2 = temp

    s = Solution()
    l3 = s.addTwoNumbers(l1.next, l2.next)
    while l3 != None:

        print(l3.val)
        l3 = l3.next
