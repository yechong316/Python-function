import os,shutil

def count_files(path):
    count = 0
    for root,dirs,files in os.walk(path):    #遍历统计
          for each in files:
                 count += 1   #统计文件夹下文件个数

    print (count)               #输出结果
def get_dir(path):  # 获取目录路径
    # 遍历path,进入每个目录都调用visit函数，
    # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
    file_paths = []
    for root, dirs, files in os.walk(path):  # 遍历path,进入每个目录都调用visit函数，，有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        print(files)
        for file in files:
            # print(dir)             #文件夹名
            file_paths.append(os.path.join(root, file))  # 把目录和文件名合成一个路径
            # print(dir)             #文件夹名
    return file_paths

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile, dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

origin_path = "D:\搜狗高速下载\图片\图片\秋瓷炫"
target_path = "D:\数据集" + '/' + '秋瓷炫'
srcfile = get_dir(origin_path)
print(srcfile)
dstfile = '/Users/xxx/tmp/tmp/1/test.sh'

for i in srcfile:
    print(i)

    mymovefile(i, target_path)