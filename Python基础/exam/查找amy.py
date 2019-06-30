students = ['jack', 'tom', 'john', 'amy', 'kim'
            , 'sunny']

# 是否找到amy的状态设定为state
state = False
for i in students:

    if i == 'amy':

        state = True
if state:
    print('find')
else:
    print('not find')