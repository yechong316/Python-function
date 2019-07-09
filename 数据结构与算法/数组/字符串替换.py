def white_str_1(str_):
    '''思路有点简单'''
    if str_ is None or str_ == '': return None

    else:

        new = []
        for i in str_:

            if i == ' ':
                new.append("%20")
            else:
                new.append(i)
                
        return ''.join(new)


def white_str_2(str_):
    '''思路有点简单'''

    num = 0
    for i in str_:

        if i == ' ':

            num += 1








if __name__ == '__main__':

    old_str = 'abd bbd'
    str_new = white_str_1(old_str)
    print(str_new)