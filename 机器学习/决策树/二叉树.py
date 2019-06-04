'''
Created on 2016/12/26
Created by freeol.cn
一些排序算法的Python实现
@author: 拽拽绅士
'''

import sys
from _ast import While
from celery.bin.celery import result

'''顺序存储的二叉树实现（非完全存储）'''
class node1(object):
    def __init__(self, S, L, R, V):
        self.S = S#自
        self.L = L#左子
        self.R = R#右子
        self.V = V#值


class tree1(object):
    def createTree(self, a):
        data = []
        for n in a:
            data.append(node1(n[0], n[1], n[2], n[3]))
        return data
    def getTree(self, a):
        return self.createTree(a)

'''链式存储的二叉树（非完全存储）'''
class tree2(object):
    def __init__(self):
        self.L = None#Left node
        self.R = None#Right node
        self.V = None#value
        self.tmp = {}
    def createTree(self, key, tree):
        if key in self.tmp:
            tmpN = self.tmp[key]
            tree.V = tmpN[3]
            Lkey = tmpN[1]
            Rkey = tmpN[2]
            if Lkey != None:
                Ltree = tree2()
                Ltree = self.createTree(Lkey, Ltree)
                tree.L = Ltree
            if Rkey != None:
                Rtree = tree2()
                Rtree = self.createTree(Rkey, Rtree)
                tree.R = Rtree
        return tree
    def getTree(self, a):
        for n in a:
            self.tmp[n[0]] = n#收集各节点信息
        tree = tree2()
        return self.createTree('1', tree)

'''判断二叉树存储结构'''
def checkTree1orTree2(tree):
    if type(tree) == list:
        return 1#顺序存储
    else:
        return 2#链式存储

'''获取根节点'''
def root(tree):
    chk = checkTree1orTree2(tree)
    if chk == 1:#顺序存储
        childKeys = {}
        for t in tree:
            if t.L != None:
                childKeys[t.L] = None
            if t.R != None:
                childKeys[t.R] = None
        for t in tree:
            if t.S not in childKeys:
                return t
    else:#链式存储
        return tree

'''获取二叉树的度'''
def degree(tree):
    chk = checkTree1orTree2(tree)
    if chk == 1:#顺序存储
        return len(tree)
    else:#链式存储
        cnt = 1
        if tree.L != None:
            cnt += degree(tree.L)
        if tree.R != None:
            cnt += degree(tree.R)
        return cnt

'''深度'''
def deepDegree(tree):
    chk = checkTree1orTree2(tree)
    if chk == 1:#顺序存储
        cnt = 0
        leafs = []#叶子集
        branchs = []#枝干集
        for t in tree:
            if t.L==None and t.R == None:
                leafs.append(t)
            else:
                branchs.append(t)
        save_cnt = 0
        for leaf in leafs:#回溯法 叶->枝->根
            cnt = 1
            key = leaf.S
            tmpBranchs = branchs.copy()
            i = 0
            while i < len(tmpBranchs):
                branch = tmpBranchs[i]
                if branch.L == key or branch.R == key:
                    cnt+=1
                    key = branch.S
                    i = 0
                else:
                    i+=1
            if cnt > save_cnt:
                save_cnt = cnt
        return save_cnt
    else:#链式存储
        cnt = 1
        Lcnt = 0
        Rcnt = 0
        if tree == None:
            return 0
        if tree.L != None:
            Lcnt = deepDegree(tree.L)
        if tree.R != None:
            Rcnt = deepDegree(tree.R)
        if Lcnt > Rcnt:
            cnt += Lcnt
        else:
            cnt += Rcnt
        return cnt

'''链式结构二叉树
前序遍历：根节点->左子树->右子树'''
def preorder(tree, m, result):
    if m == 1:#非递归实现(栈)
        static = []#栈
        t = tree
        '''法1
        while t != None or static != []:
            while t != None:
                result.append(t)
                static.append(t)
                t=t.L
            if static != []:
                t = static.pop()
                t = t.R
        '''
        static.append(tree)
        while static:
            n = static.pop()
            result.append(n)
            if n.R:
                static.append(n.R)
            if n.L:
                static.append(n.L)

    else:#递归实现
        if tree == None:
            return result
        result.append(tree)
        result=preorder(tree.L, 2, result)
        result=preorder(tree.R, 2, result)
    return result

'''链式结构二叉树
中序遍历：左子树->根节点->右子树'''
def inorder(tree, m, result):
    if m == 1:#非递归实现(栈)
        static = []#栈
        t = tree
        '''法1
        while t != None or static != []:
            while t != None:
                static.append(t)
                t=t.L
            if static != []:
                t = static.pop()
                result.append(t)
                t = t.R
        '''
        while t != None or static != []:
            while t != None:
                static.append(t)
                t = t.L
            t = static.pop()
            result.append(t)
            t = t.R
    else:#递归实现
        if tree == None:
            return result
        result=inorder(tree.L, 2, result)
        result.append(tree)
        result=inorder(tree.R, 2, result)
    return result

'''链式结构二叉树
后序遍历：左子树->右子树->根节点'''
def postorder(tree, m, result):
    if m == 1:#非递归实现(栈)
        static = []#栈
        t = tree
        mk = None
        while t != None or static != []:
            while t != None:
                static.append(t)
                t = t.L
            t = static.pop()
            if t.R == None or t.R == mk:
                result.append(t)
                mk = t
                t = None
            else:
                static.append(t)
                t = t.R
    else:#递归实现
        if tree == None:
            return result
        result = postorder(tree.L, 2, result)
        result = postorder(tree.R, 2, result)
        result.append(tree)
    return result

'''order value print'''
def resultPrintV(msg, rs):
    v=[]
    for t in rs:
        v.append(t.V)
    print(msg, v)


'''期望高度'''


def main():
    '''    1
          ∧
        2    3
       ∧    ∧
      4  5  9  7
    ∧   ∧
   8 6 10 11'''
    data = [ #原始数据
            ['1', '2', '3', 1],#Self key, Left key, Right key, Value
            ['2', '4', '5', 2],
            ['3', '9', '7', 3],
            ['4', '8', '6', 4],
            ['5', '10', '11', 5],
            ['9', None, None, 9],
            ['7', None, None, 7],
            ['8', None, None, 8],
            ['6', None, None, 6],
            ['10', None, None, 10],
            ['11', None, None, 11],
           ]
    print('原始数据大小', sys.getsizeof(data))
    print('预计二叉树根节点值', 1)
    print('预计二叉树的度', 11)
    print('预计二叉树的深度', 4)
    print('预计前序遍历值的结果', [1, 2, 4, 8, 6, 5, 10, 11, 3, 9, 7])
    print('预计中序遍历值的结果', [8, 4, 6, 2, 10, 5, 11, 1, 9, 3, 7])
    print('预计后序遍历值的结果', [8, 6, 4, 10, 11, 5, 2, 9, 7, 3, 1])

    print('========>创建顺序结构二叉树')
    t1 = tree1().getTree(data)#顺序结构
    print('顺序结构二叉树大小', sys.getsizeof(t1))
    root1 = root(t1)
    print('顺序结构二叉树根节点值', root1.V)
    print('顺序结构二叉树的度', degree(t1))
    print('顺序结构二叉树的深度', deepDegree(t1))

    print('========>创建链式结构二叉树')
    t2 = tree2().getTree(data)#链式结构
    print('链式结构二叉树大小', sys.getsizeof(t2))
    root2 = root(t2)
    print('链式结构二叉树根节点值', root2.V)
    print('链式结构二叉树的度', degree(t2))
    print('链式结构二叉树的深度', deepDegree(t2))
    rs = [];resultPrintV('链式结构 前序遍历值的结果->非递归实现', preorder(t2, 1, rs))
    rs = [];resultPrintV('链式结构 前序遍历值的结果->递归实现', preorder(t2, 2, rs))
    rs = [];resultPrintV('链式结构 中序遍历值的结果->非递归实现', inorder(t2, 1, rs))
    rs = [];resultPrintV('链式结构 中序遍历值的结果->递归实现', inorder(t2, 2, rs))
    rs = [];resultPrintV('链式结构 后序遍历值的结果->非递归实现', postorder(t2, 1, rs))
    rs = [];resultPrintV('链式结构 后序遍历值的结果->递归实现', postorder(t2, 2, rs))

if __name__ == '__main__':
    main()