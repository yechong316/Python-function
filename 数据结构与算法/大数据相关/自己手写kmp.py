

def KMP(string, substring):

    prefix = get_prefix(substring)

    m, n = len(string), len(substring)
    i, j = 0, 0
    while i < m and j < n:

        if string[i] == substring[j]:

            i += 1
            j += 1
        elif j != 0:
            j = prefix[j - 1]

        else:

            j = 0
            i += 1

    if j == n:

        return i - j
    else:
        return -1








def get_prefix(substring):

    index, m = 0, len(substring)

    prefix = [0] * m
    i = 1
    while i < m:

        if substring[i] == substring[index]:

            prefix[i] = index
            index += 1
            i += 1
        elif index != 0:
            index = prefix[index - 1]
        else:

            prefix[index] = 0
            i += 1


    return prefix

prefix = KMP('dafagagq', 'q')
print(prefix)
