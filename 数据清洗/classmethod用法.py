#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
classmethod这个东西其实没有什么，无非就是不需要传入self函数，但是第一个参数必须是cls
'''



class A(object):
    bar = 1
    def func1(self):  
        print ('foo') 
    @classmethod
    def func2(cls):
        print ('func2')
        print (cls.bar)
        cls().func1()   # 调用 foo 方法
 
A.func2()  