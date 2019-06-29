import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # 如果是空图 返回没有陆地
        if not grid:
            return 0
        # 把原图改变一下  四周加上一圈"0" 防止出界 方便遍历
        w, h = len(grid[0]), len(grid)
        new_grid = [[0 for i in range(w + 2)]]
        for g in grid:
            new_grid.append([0] + g + [0])
        new_grid.append([0 for i in range(w + 2)])

        num = 0  # 记录陆地数量

        # 遍历除了周围的"0" 中间的部分
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if new_grid[i][j] == 1:  # 如果是陆地 就进入深度遍历
                    num += 1
                    self.deep_search(i, j, new_grid)

        return num

    def deep_search(self, i, j, grid):
        """如果当前是陆地，把当前结点标记遍历过，并分别看左右上下四个位置"""
        if grid[i][j] == 0:
            return
        grid[i][j] = 0
        self.deep_search(i - 1, j, grid)
        self.deep_search(i, j - 1, grid)
        self.deep_search(i, j + 1, grid)
        self.deep_search(i + 1, j, grid)

s = Solution()
m, n = 6, 6
grid = [[
    np.random.randint(0, 2) for i in range(n)
] for _ in range(n)]
print(grid)
num = s.numIslands(grid)
print('岛屿数量', num)
plt.imshow(grid)
plt.show()