# Definition for singly-linked list.

def list2LinkList(alist):
    '''list --> Link List'''
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None
    node_list = [
        ListNode(i) for i in alist
    ]

    for i in range(len(node_list) - 1):

        node_list[i].next = node_list[i + 1]

    return node_list


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int):
        
        
        length = self.length(head)
        left_point, right_point = 1, length
        left_node, right_node = head, self.find_final_node(head)
        
        while left_point != m:
            left_node = left_node.next
            left_point += 1
        
        while right_point != n:
            right_node = right_node.next
            right_point -= 1
        
        while left_point <= right_point:
            left_node.val, right_node.val = right_node.val, left_node.val

        return head



        pass
    def length(self, head):
        count = 1
        current_node = head

        while current_node != None:
            count += 1
            current_node = current_node.next
        # print(count)
        return count
    
    def find_final_node(self, head):
        
        cur = head
        while cur.next != None:
            
            cur = cur.next
        return cur


if __name__ == '__main__':

    a = [1, 2, 3, 4, 6]
    node_a = list2LinkList(a)

    ss = Solution()
    ss_rever = ss.reverseBetween(node_a[0], 2, 4)

    cur = ss_rever
    while cur.next != None:
        print(cur.val)
        cur = cur.next

