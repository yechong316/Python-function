class Node:

    def __init__(self, ele):

        self.ele = ele # dytpe: float
        self.lchild = None # dytpe: node
        self.rchild = None # dytpe: node

    def travel(self):
        print(self.ele, end=' ')
        
        if self.lchild is not None:
            print(self.lchild.ele, end=' ')
        else:
            return
        if self.rchild is not None:
            print(self.rchild.ele, end=' ')
        else:
            return

class Tree:


    def __init__(self, node=None):
        self.root = node

    def add(self, ele):

        node = Node(ele)
        if self.root is None:
            self.root = node
            return
        
        quene = [self.root]

        while quene:
            cur = quene.pop(0)
            if cur.lchild is None:
                cur.lchild = node
                return

            else:
                quene.append(cur.lchild)

            if cur.rchild is None:
                cur.rchild = node
                return

            else:

                quene.append(cur.rchild)


    def delete(self, ele):
        pass

    def modefied(self, ele):
        pass

    def depth(self):
        pass

    def travel(self):

        if self.root is None:return
        quene = [self.root]
        while quene:
            cur = quene.pop(0)
            print(cur.ele, end=' ')
            if cur.lchild is not None:quene.append(cur.lchild)
            if cur.rchild is not None:quene.append(cur.rchild)

    def preorder(self, node):

        if node is None: return

        print(node.ele, end=' ')
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def midorder(self, node):

        if node is None: return

        self.preorder(node.lchild)
        print(node.ele, end=' ')
        self.preorder(node.rchild)

    def backorder(self, node):

        if node is None: return

        self.preorder(node.lchild)
        self.preorder(node.rchild)
        print(node.ele, end=' ')


if __name__ == '__main__':

    tree = Tree()
    tree.add(1)
    tree.add(2)
    tree.add(3)
    tree.add(4)
    # tree.add(5)
    # tree.add(6)
    # tree.add(7)
    # tree.add(8)
    # tree.travel()
    print('')
    tree.preorder(tree.root)
    print('')
    tree.midorder(tree.root)
    print('')
    tree.backorder(tree.root)
