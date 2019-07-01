class Solution(object):
    def hasCycle(self, head):
        if not head:
            return False
        p1 = head
        p2 = head.next
        while 1:
            if p1 == None or p2 == None or p2.next == None:
                return False
            elif p1 == p2:
                return True
            else:
                p1 = p1.next
                p2 = p2.next.next