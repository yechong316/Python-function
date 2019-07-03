def has_circle(head):

    if head is None or head.next is None: return

    else:

        p1 = head
        p2 = head.next

        while 1:

            if p1 == None or p2 == None or p2.next == None: return False # 考虑p2.next == None是为了防止索引越界

            elif p1 == p2: return True
            else:

                p1 = p1.next
                p2 = p2.next.next


if __name__ == '__main__':

    # from single_linked_list import *
    # pass
    pass


