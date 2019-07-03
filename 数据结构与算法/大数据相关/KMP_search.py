def KMP_algorithm(string, substring):
    '''
    KMP字符串匹配的主函数
    若存在字串返回字串在字符串中开始的位置下标，或者返回-1
    '''
    pnext = gen_pnext(substring)
    n = len(string)
    m = len(substring)
    i, j = 0, 0
    while i < n and j < m:

        # 如果两个字符串相同，则i，j两个指针同时往右移动
        if string[i] == substring[j]:
            i += 1
            j += 1
        # 如果在j=0，对-1索引会索引到最后一个元素
        elif j != 0:
            j = pnext[j - 1]
        else:
            i += 1
    if j == m:
        return i - j
    else:
        return -1


def gen_pnext(substring):
    """
    构造临时数组pnext
    """
    index, m = 0, len(substring)
    pnext = [0] * m

    # i 代表prefix里面的索引号
    i = 1
    while i < m:
        if (substring[i] == substring[index]):
            pnext[i] = index + 1
            index += 1
            i += 1
        elif (index != 0):
            index = pnext[index - 1]
        else:
            pnext[i] = 0
            i += 1
    return pnext


if __name__ == "__main__":
    string = 'abcxabcdabcdabcy'
    substring = 'abad'
    out = KMP_algorithm(string, substring)
    print(out)