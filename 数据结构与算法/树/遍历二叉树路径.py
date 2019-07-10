"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    # @param {TreeNode} root the root of the binary tree
    # @return {List[str]} all root-to-leaf paths
    def binaryTreePaths(self, root):
        # Write your code here
         # Write your code here
        if root is None:
            return []
        stack = [root]#将root节点放入栈里面
        result = []
        while len(stack) != 0:#栈不为空就循环操作
            topnode = stack[-1] #读取栈顶元素，但是将元素弹出
            #从栈顶节点开始遍历子节点。直到寻找到叶子节点位置
            while topnode.left is not None or topnode.right is not None:
                if topnode.left is not None:#寻找过程中优先寻找位于左子树的节点
                    stack.append(topnode.left)
                    p = topnode.left
                    topnode.left = None #入栈后要置空，否则出栈的条件无法判断，或者可以寻找其他的办法标记哪些节点已经入过栈。
                    topnode = p
                elif topnode.right is not None:
                    stack.append(topnode.right)
                    p = topnode.right
                    topnode.right = None
                    topnode = p
            path = ""
            #到这里表示已经寻找完一条路径，拼接字符串
            for idx,node in enumerate(stack):
                path += str(node.val)
                if idx != len(stack)-1:
                    path += "->"
            result.append(path)
            #将最后一个节点出栈。一直寻找到没有入栈的节点。
            while topnode.left is None and topnode.right is None:
                stack.pop()
                if len(stack) == 0:
                    return  result
                topnode = stack[-1]
        return result