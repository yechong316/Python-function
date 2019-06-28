# 从数字中找到重复数字
import numpy as np
def find_duplication(a_list):

    if a_list is None and len(a_list) == 1:
        return False
    else:
        for i in range(len(a_list)):

            if a_list[i] == i:
                continue
            elif a_list[i] == a_list[a_list[i]]:
                # print(a_list[i])
                return True
            else:
                a_list[i], a_list[a_list[i]] = a_list[a_list[i]], a_list[i]



if __name__ == '__main__':

    n = 100
    a = np.random.randint(0, n, size=n)
    print('数组为：', a)
    print(find_duplication(a))