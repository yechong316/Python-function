'''

利用set自动去重
'''

if __name__ == '__main__':

    a = set()

    for _ in range(100):

        a.add(1)
        a.add(3)

    b = set()
    b.add(3)
    b.add(5)

    # 对两个集合取交集
    print(b&a)

    # 对两个集合取并集
    print(b|a)