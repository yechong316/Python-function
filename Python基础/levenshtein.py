# -*- coding: utf-8 -*-


import numpy as np


def distance_str(str1, str2):

    dp = np.zeros((len(str1) + 1, len(str2) + 1))
    m = len(str1)
    n = len(str2)
    for k in range(1, m + 1):
        dp[k][0] = k
    for k in range(1, n + 1):
        dp[0][k] = k
    for k in range(1, m + 1):
        for j in range(1, n + 1):
            dp[k][j] = min(dp[k - 1][j], dp[k][j - 1]) + 1  # 这里表示上边和下边的数值最小数值
            if str1[k - 1] == str2[j - 1]:
                dp[k][j] = min(dp[k][j], dp[k - 1][j - 1])
            else:
                dp[k][j] = min(dp[k][j], dp[k - 1][j - 1] + 1)
    print(int(dp[-1][-1]))


if __name__ == '__main__':
    str1 = "eeba"
    str2 = "abac"

    distance_str(str1, str2)
