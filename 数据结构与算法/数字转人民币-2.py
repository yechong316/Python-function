def num2rmb(value):

    s = str(value)

    length = len(s)

    if length > 10:
        print('数字太大，无法处理')
        return

    num = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    iunit = ['十', '百', '千', '万', '十', '百', '千', '亿']

    so = []
    for i, n in enumerate(s):

        if i == 4:
            so.append(num[n])
            so.append(num[n])





