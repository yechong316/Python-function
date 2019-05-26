import copy


if __name__ == '__main__':

    a = [1, 2, 3, ['f', 'a']]
    d = a
    b = copy.copy(a)
    c = copy.deepcopy(a)
    # print('原始数据:', a)
    # print('赋值', d)
    # print('普通复制数据:', b)
    # print('深copy:', b)

    a.append(4)
    a[3].append('e')
    # print('a添加数据后')
    print('原数据', a)
    print('赋值：', d)
    print('浅拷贝', b)
    print('深拷贝', c)