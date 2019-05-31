'''
约瑟夫问题:
一个一世纪著名历史学家弗拉维奥·约瑟夫斯的传奇故事。故事讲的是，他和他的 39
个战友被罗马军队包围在洞中。他们决定宁愿死，也不成为罗马人的奴隶。他们围成
一个圈，其中一人被指定为第一个人，顺时针报数到第七人，就将他杀死。约瑟夫斯
是一个成功的数学家，他立即想出了应该坐到哪才能成为最后一人。最后，他加入了
罗马的一方，而不是杀了自己。
'''

# 解决思路，将士兵组合为循环链表，循环遍历，计数器等于7就删除当前节点，直至当前链表长度为1
from 线性表.double_recurrent_LL import *


def Josef(list, num=7):


    ll = Double_link_list()
    ll.list2ll(list)

    length = ll.length()

    count = 0
    cur = ll._head
    first = ll._head
    while ll.length() > 1:

        last_node = ll.final_node()
        # 定义首节点

        # 开始遍历节点
        cur = cur.next
        count += 1

        if count == num:
            count = 1
            # 当被删除节点是链表首节点时，需要将末尾节点的指针指向第二个节点，_head指向第二个节点
            if cur == ll._head:


                next_node = cur.next

                ll._head = next_node
                next_node.pre = ll._head
                last_node.next = next_node
                print('链表长度为{}, 删除节点号码：{}'.format(ll.length(), cur.ele))

            elif cur == last_node:
                
                pre_node = cur.pre

                pre_node.next = last_node
                print('链表长度为{}, 删除节点号码：{}'.format(ll.length(), cur.ele))

            else:

                pre_node = cur.pre
                next_node = cur.next

                pre_node.next = next_node
                next_node.pre = pre_node
                print('链表长度为{}, 删除节点号码：{}'.format(ll.length(), cur.ele))

        length = ll.length()


    return ll._head










a = list(range(1, 38))

ysf_node = Josef(a)

print('最佳的人选:', ysf_node.ele)
