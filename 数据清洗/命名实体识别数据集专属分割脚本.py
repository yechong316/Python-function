# encoding=utf8


import os
import pickle
import random
from tqdm import tqdm




def get_dir(path):  # 获取目录路径
    # 遍历path,进入每个目录都调用visit函数，
    # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
    file_paths = []

    files_name = list(os.walk(path))[0][2]

    file_paths = [
        os.path.join(path, i) for i in files_name
    ]
    return file_paths, files_name



def single_text_samples(source_path, target_path, sample=0.01):


    with open(target_path, 'w', encoding='utf-8') as f_target:

        with open(source_path, 'r', encoding='utf-8') as f_source:


            l_total = [[]]

            lines = f_source.readlines()
            for line in tqdm(lines):

                if line == '\n':

                    l_total[-1].append('\n')
                    l_total.append([])
                    continue

                l_total[-1].append(line)
            count = 1
            for i in l_total:

                count += 1
                if count > sample:
                    break
                for j in i:
                    f_target.write(j)



    print('文本切割成功，储存至{}，采样{}条数据'.format(target_path, sample))

if __name__ == '__main__':

    data_dir = './原始数据'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file_paths, files = get_dir(data_dir)

    for i,j in zip(file_paths, files):

        single_text_samples(i, j, 50)

