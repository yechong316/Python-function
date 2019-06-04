class Node:

    def __init__(self, id, split, depth):

        self.id = id
        self.split = split
        self.depth = depth
        if self.id == 0:
            print('我就是根节点')
        else:
            print('我的父ID号是{}:'.format(self.id))
        # print('我的分类依据是{}:'.format(self.split))
        # print('我在第{}层'.format(self.depth))


class Tree:



    def add_child(self, father_id, left, right):
        self.father_id = father_id
        self.left = left
        self.right = right
        print('{} --->{}'.format(self.father_id, self.left))
        print('{} --->{}'.format(self.father_id, self.right))

Node1 = Node(0, {'feature':'vel','value':0.350}, 0)
Node2 = Node(1, {'feature':'lateral_axis','value':-0.316}, 1)
Node3 = Node(1, {'feature':'RSSI','value': -51.25}, 1)


tree = Tree()
tree.add_child(Node1.id, Node2.id, Node3.id)