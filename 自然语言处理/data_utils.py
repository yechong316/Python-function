# -*- coding: utf-8 -*
'''



2019年6月12日15:00:21
适用条件，已经由语料库，并进行数据处理，得到两个变量，data，和labels，data里面是m个样本，每个样本有n个字/词
label同样是m个样本，对应的n个标签，开发初期考虑用在命名实体识别项目，词性标注等方面
:param datas: 【样本数，汉字个数】,请提前转换为numpy格式
:param labels:【样本数，label个数】,请提前转换为numpy格式
:param buckket_gap: 分桶间隔，按照这个数值分一个桶，默认为10
:param max_seq_length: 超过最大句子长度的自动截去，保留前面的内容，默认为50
:param min_seq_length: 若语料库中最小句子长度大于这个值+gap的话，那么从句子的最小长度开始分桶
'''
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import pickle


class BucketData:

    def __init__(self, datas, buckets):

        '''
        新版 2019年6月12日17:00:13
        :param datas:
        :param buckets: 根据桶的最后一个列表里面的最后一个元素取为最大长度
        '''

        self.datas = datas
        self.buckets = buckets

        # 分桶 + PAD
        self.generate_bucket()

        print('实例化完毕')

    def generate_bucket(self):
        '''
        后期增加参数，自动判断，句子最短长度为
        :return:
        '''

        self.n_buckets = len(self.buckets)
        self.buckets_data = [[] for _ in range(self.n_buckets)]

        pad = 0
        for i in range(len(self.datas)):

            # 获取当前句子的长度
            cur_length = len(self.datas[i])

            # 遍历分桶尺寸
            for j in range(self.n_buckets):

                # 计算当前分桶的最大最小长度
                min_length, max_length = self.buckets[j][0], self.buckets[j][1]

                if cur_length > min_length and cur_length <= max_length:

                    self.datas[i].extend([pad] * (max_length - cur_length))
                    self.buckets_data[j].append(self.datas[i])

    def random(self):
        # while True:
        #     # 选择一个[1, MAX(ROWID)]中的整数，读取这一行
        #     rowid = np.random.randint(1, self.size + 1)
        pass


if __name__ == '__main__':
    pkl_path = r'datas_labels_full'

    buckets = [
        [2, 10], [10, 20], [20, 30], [30, 40]
    ]
    with open(pkl_path, 'rb') as inp:
        datas = pickle.load(inp)
        labels = pickle.load(inp)

    bucket_datas = BucketData(datas, buckets)
    bucket_labels = BucketData(labels, buckets)
    

    # number = np.random.sample()

