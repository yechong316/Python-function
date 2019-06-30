def compare(op1, op2):
    """
    比较两个运算符的优先级,乘除运算优先级比加减高
    op1优先级比op2高返回True，否则返回False
    """
    return op1 in ["*", "/"] and op2 in ["+", "-"]


def getvalue(num1, num2, operator):
    """
    根据运算符号operator计算结果并返回
    """
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    else:  # /
        return num1 / num2


def process(data, opt):
    """
    opt出栈一个运算符，data出栈两个数值，进行一次计算，并将结果入栈data
    """
    operator = opt.pop()
    num2 = data.pop()
    num1 = data.pop()
    data.append(getvalue(num1, num2, operator))


def calculate(s):
    """
    计算字符串表达式的值,字符串中不包含空格
    """
    data = []  # 数据栈
    opt = []  # 操作符栈
    i = 0  # 表达式遍历索引
    while i < len(s):
        if s[i].isdigit():  # 数字，入栈data
            start = i  # 数字字符开始位置
            while i + 1 < len(s) and s[i + 1].isdigit():
                i += 1
            data.append(int(s[start: i + 1]))  # i为最后一个数字字符的位置
        elif s[i] == ")":  # 右括号，opt出栈同时data出栈并计算，计算结果入栈data，直到opt出栈一个左括号
            while opt[-1] != "(":
                process(data, opt)
            opt.pop()  # 出栈"("
        elif not opt or opt[-1] == "(":  # 操作符栈为空，或者操作符栈顶为左括号，操作符直接入栈opt
            opt.append(s[i])
        elif s[i] == "(" or compare(s[i], opt[-1]):  # 当前操作符为左括号或者比栈顶操作符优先级高，操作符直接入栈opt
            opt.append(s[i])
        else:  # 优先级不比栈顶操作符高时，opt出栈同时data出栈并计算，计算结果如栈data
            while opt and not compare(s[i], opt[-1]):
                if opt[-1] == "(":  # 若遇到左括号，停止计算
                    break
                process(data, opt)
            opt.append(s[i])
        i += 1  # 遍历索引后移
    while opt:
        process(data, opt)
    print(data.pop())

