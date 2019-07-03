# coding=utf8
from collections import deque




# 递归深度优先
# 深度优先是一次访问到底，知道没有连接的顶点，再逐级返回，再逐级访问
def dfs(G, v, visited=set()):
    print(v)
    visited.add(v)  # 用来存放已经访问过的顶点
    # G[v] 是这个顶点的相邻的顶点
    for u in G[v]:
        # 这一步很重要，否则就进入了无限循环，只有这个顶点没有出现在这个集合中才会访问
        if u not in visited:
            dfs(G, u, visited)


# 迭代深度优先
def dfs_iter(G, v):
    visited = set()
    s = [v]
    while s:
        u = s.pop()
        if u not in visited:
            print(u)

            visited.add(u)
            s.extend(G[u])





# 广度优先遍历
# 广度优先是，从某一个顶点出发，先访问一步能够到达的顶点，再访问两步能够达到的顶点，以此类推
def bfs(G, v):
    q = deque([v])
    # 同样需要申明一个集合来存放已经访问过的顶点，也可以用列表
    visited = {v}
    while q:
        u = q.popleft()
        print(u)

        for w in G[u]:
            if w not in visited:
                q.append(w)
                visited.add(w)


#

if __name__ == '__main__':
    # 深度优先遍历
    G = [
        {1, 2, 3},  # 0
        {0, 4, 6},  # 1
        {0, 3},  # 2
        {0, 2, 4},  # 3
        {1, 3, 5, 6},  # 4
        {4, 7},  # 5
        {1, 4},  # 6
        {5, }
    ]

    print(G)
    dfs(G, 0)
    # dfs_iter(G, 0)

    # print("BFS")
    # bfs(G, 0)