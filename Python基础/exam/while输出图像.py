i = 0
length = 10
while i < length:
    i += 1
    num = int(length/2 - abs(length/2 - i))
    print('*'*num)
