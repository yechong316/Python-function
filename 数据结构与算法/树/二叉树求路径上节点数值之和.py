"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    # @param {TreeNode} root the root of binary tree
    # @param {int} target an integer
    # @return {int[][]} all valid paths
    def __init__(self):
        self.ans = []
        self.s = []

    # 递归解法
    def binaryTreePathSum(self, root, target):
        # Write your code here
        if root is None:
            return self.ans

        self.s.append(root.val)
        target -= root.val

        if target == 0 and root.left is None and root.right is None:
            self.ans.append(list(self.s))

        self.binaryTreePathSum(root.left, target)
        self.binaryTreePathSum(root.right, target)

        self.s.pop()
        return self.ans

    # 非递归解法
    def binaryTreePathSum1(self, root, target):
        # Write your code here
        s = []
        res = []
        top = -1
        sum = 0
        p = root
        while True:
            while p is not None:
                sum += p.val
                s.append(p)
                top += 1
                p = p.left
            f = True
            q = None
            while top != -1 and f :
                p = s[-1]
                if p.right == q:
                    if p.right is None and p.left is None and sum == target:
                        t = []
                        for i in s:
                            t.append(i.val)
                        res.append(t)
                    sum -= p.val
                    top -= 1
                    s.pop()
                    q = p
                else:
                    p = p.right
                    f = False
            if top == -1:
                break
        return res