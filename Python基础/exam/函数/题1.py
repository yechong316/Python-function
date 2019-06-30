def caculate(num1, num2, opt):

    if opt == '+':

        return num1 + num2
    elif opt == '-':
        return num1 - num2
    elif opt == '*':
        return num1 * num2
    elif opt == '/':
        return num1 / num2

if __name__ == '__main__':

    num1, num2, opt = 10, 20, '-'
    result = caculate(num1, num2, opt)
    print(result)