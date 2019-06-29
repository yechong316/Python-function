from queue import Queue
import numpy as np
import matplotlib.pyplot as plt


class Solution(object):
    def numIslands(self, grid):
        try:
            r = 0
            m = len(grid)
            n = len(grid[0])
            around = ((0, 1), (1, 0), (0, -1), (-1, 0))
        except:
            return 0

        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    r += 1

                    # ---------------------------BFS 开始-----------------------------
                    # 1.把根节点投入队列
                    q = Queue()
                    q.put((i, j))

                    # 开始循环
                    while not q.empty():
                        # 取出还未沉没的陆地节点并沉没陆地（防止下次遍历到的时候再算一遍）
                        x, y = q.get()

                        if int(grid[x][y]):
                            grid[x][y] = '0'

                            # 放入周围的陆地节点
                            for a, b in around:
                                a += x
                                b += y
                                if 0 <= a < m and 0 <= b < n and int(grid[a][b]):  # 判断越界和类型
                                    q.put((a, b))
                    # ----------------------------------------------------------------
        return r
m, n = 6, 6
grid = [[
    np.random.randint(0, 2) for i in range(n)
] for _ in range(n)]
plt.imshow(grid)
s = Solution()
num = s.numIslands(grid)
print(num)

plt.show()