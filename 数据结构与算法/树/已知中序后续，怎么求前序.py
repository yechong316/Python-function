def last_sort(str1, str2):
    if len(str2) <= 1:
        return str2
    else:

        str_1 = last_sort(str1[1:str2.index(str1[0])+1], str2[:str2.index(str1[0])])
        str_2 = last_sort(str1[str2.index(str1[0])+1:], str2[str2.index(str1[0])+1:])
        str_3 = str1[0:1]
        return str_1 + str_2 + str_3


str1 = ['A', 'B', 'D', 'C', 'E', 'F']
str2 = ['D', 'B', 'A', 'E', 'C', 'F']
print(last_sort(str1, str2))