# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def middleNode(self, head):

        length = self.length(head)
        # print('length:', length)
        middle = length // 2
        cur = head
        count = 0

        while count != middle:
            cur = cur.next
            count += 1

        return cur

    def length(self, head):
        count = 1
        current_node = head

        while current_node != None:
            count += 1
            current_node = current_node.next
        # print(count)
        return count

a = [1, 2, 3, 4]


node_list = [
    ListNode(i) for i in a
]

for i in range(len(node_list) - 1):

    node_list[i].next = node_list[i + 1]
head = ListNode(None)
head.next = node_list[0]
b = Solution()
value = b.middleNode(head)
print(value.val)