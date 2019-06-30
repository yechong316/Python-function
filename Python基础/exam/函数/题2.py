import pickle

student_sys = {}
account = input('请输入账户名：', )
passwords = input('请输入账户密码：', )
state = False
if account == 'admin' and passwords == 123456:

    state = True
def add_student(name):

    if state:
        student_sys['name'] = name

def delete_student(name):
    if state:
        del student_sys[name]
    
def modefied(name):
    if state:
        student_sys['name'] = name
    
def search(name):
    if state:
        if name in student_sys:
            print('该学员在系统中！')
        else:
            print('该学员不在系统中！')

def logout_sys():

    # 访问状态关闭
    return False