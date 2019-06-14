'''
摘自菜鸟教程 ------https://www.runoob.com/python3/python3-string-startswith.html
描述
startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查。

语法
startswith()方法语法：

str.startswith(substr, beg=0,end=len(string));
参数
str -- 检测的字符串。
substr -- 指定的子字符串。
strbeg -- 可选参数用于设置字符串检测的起始位置。
strend -- 可选参数用于设置字符串检测的结束位置。
返回值
如果检测到字符串则返回True，否则返回False。
'''

#!/usr/bin/python3
 
str = "this is string example....wow!!!"
print (str.startswith( 'this' ))   # 字符串是否以 this 开头
print (str.startswith( 'string', 8 ))  # 从第八个字符开始的字符串是否以 string 开头
print (str.startswith( 'this', 2, 4 )) # 从第2个字符开始到第四个字符结束的字符串是否以 this 开头











