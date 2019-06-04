import sqlite3
import os
db_path = 'test.db'

if os.path.exists(db_path):
    os.remove(db_path)
# 建立连接
con = sqlite3.connect(db_path)

# 1、游标对象的使用
cur = con.cursor()

# 、创建表
cur.execute('CREATE TABLE person(id integer primary key, name varchar(20),age integer)')


# 插入数据
for _ in range(10):

    data = "{}, 'leon_{}', {}".format(_, _, _)
    cur.execute('INSERT INTO person VALUES (%s)'%data)

# 这两种插入数据操作都不会立即生效，需要使用数据库对象con进行提交操作
con.commit()

# 查询数据
cur.execute('SELECT * FROM person')

res = cur.fetchall()

for line in res:
    print('每行数据的值>>', line)


cur.execute('UPDATE person name=?WHERE ID')
cur.execute('DELETE FROM person WHERE id=1')
con.commit()

cur.execute('SELECT * FROM person')
res = cur.fetchall()

for line in res:
    print('每行数据的值>>', line)







