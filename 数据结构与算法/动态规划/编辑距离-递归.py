def lev(str_1, str_2):

    if str_1 is None or str_2 is None: return


    if len(str_1) == 0: return len(str_2)
    elif len(str_2) == 0: return len(str_1)
    elif str_1 == str_2 : return 0

    if str_1[len(str_1) - 1] == str_2[len(str_2) - 1]:
        d = 0
    else:
        d = 1


    dele_length = lev(str_1[:-1], str_2) + 1
    replace_length = lev(str_1, str_2[:-1]) + 1
    insert_length = lev(str_1[:-1], str_2[:-1]) + d

    return min(dele_length, replace_length, insert_length)
str1 = 'abce'
str2 = 'aaced'


l = lev(str1, str2)
print(l)
