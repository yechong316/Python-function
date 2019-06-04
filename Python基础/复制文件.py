# -- encoding:utf-8 --
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import numpy as np
import datetime
import os,shutil
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        pass
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        # print('move %s -> %s"%( srcfile,dstfile))
#         dh
def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        pass
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件


job_names = mdb.jobs.keys()
for i in job_names:
    srcfile = 'd:/Temp/' + i + '.odb'
    dstfile = "D:\旧电脑\[2019年5月7日]固体发动机报告\obd文件\结构01/" + i + '.odb'
    mymovefile(srcfile, dstfile)