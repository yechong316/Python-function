import numpy as np
import matplotlib.pyplot as plt



class Solution:

    def numIslands(self, grid):

        # 对grid先进行padding
        m, n = len(grid[0]), len(grid)

        new_grid = [[0 for _ in range(m + 2)]]

        for g in grid:

            new_grid.append([0] + g + [0])
        new_grid.append([0 for _ in range(m + 2)])

        # 初始岛屿数量
        num = 0

        # 开始遍历
        for i in range(1, n + 1):

            for j in range(1, m + 1):

                if new_grid[i][j] == 1:
                    # 岛屿数量直接+1
                    num += 1

                    self.search_near_region(i, j, new_grid)
        return num

    def search_near_region(self, i, j, grid):

        if grid[i][j] == 0:

            # 海洋直接返回
            return
        else:

            # 搜索过的区域标记为0
            grid[i][j] = 0

            # 对岛屿周边进行搜索
            self.search_near_region(i + 1, j, grid)
            self.search_near_region(i - 1, j, grid)
            self.search_near_region(i, j + 1, grid)
            self.search_near_region(i, j - 1, grid)


s = Solution()
m, n = 4, 4
grid = [[
    np.random.randint(0, 2) for i in range(n)
] for _ in range(n)]
print(grid)
num = s.numIslands(grid)
print('岛屿数量', num)
plt.imshow(grid)
plt.show()