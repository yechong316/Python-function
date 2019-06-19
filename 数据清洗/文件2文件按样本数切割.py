# encoding=utf8
import os
import pickle
import random
from tqdm import tqdm

# with open('train_dev_test_sentences', 'rb') as f:
#
#     train_sentences = pickle.load(f)
#     dev_sentences = pickle.load(f)
#     test_sentences = pickle.load(f)

data_dir = './原始数据'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
data_train = os.path.join(data_dir, 'train.tsv')
data_dev   = os.path.join(data_dir, 'dev.tsv')
data_test  = os.path.join(data_dir, 'test.tsv')

def single_text_samples(source_path, target_path, sample=0.01):


    with open(target_path, 'w', encoding='utf-8') as f_target:

        with open(source_path, 'r', encoding='utf-8') as f_source:

            count = 1
            while True:

                lines = f_source.readline()
                f_target.write(lines)
                count += 1

                if count >= sample:

                    break




    print('文本切割成功，储存至{}，采样{}条数据'.format(target_path, sample))


single_text_samples(data_train, 'train.tsv', 20)
single_text_samples(data_dev , 'dev.tsv', 10)
single_text_samples(data_test, 'test.tsv', 10)
