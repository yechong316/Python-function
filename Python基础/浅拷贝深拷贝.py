import copy


if __name__ == '__main__':

    a = [1, 2, 3, ['f', 'a']]
    print('原数据', id(a))
    print('开始修改...')
    d = a
    b = copy.copy(a)
    print('浅拷贝', id(b))
    c = copy.deepcopy(a)
    # print('原始数据:', a)
    # print('赋值', d)
    # print('普通复制数据:', b)
    # print('深copy:', b)

    a.append(4)
    a[3].append('e')
    # print('a添加数据后')
    print('原数据', id(a))
    print('赋值：', id(d))
    print('浅拷贝', id(b))
    print('浅拷贝', b)
    print('深拷贝', id(c))