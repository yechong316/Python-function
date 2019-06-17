# encoding=utf8
import os
import pickle
import random
from tqdm import tqdm

with open('train_dev_test_sentences', 'rb') as f:

    train_sentences = pickle.load(f)
    dev_sentences = pickle.load(f)
    test_sentences = pickle.load(f)

data_dir = './data_samples'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
data_train = os.path.join(data_dir, 'example.train')
data_dev   = os.path.join(data_dir, 'example.dev')
data_test  = os.path.join(data_dir, 'example.test')

def single_text_samples(text_list, target_path, sample=0.01):


    with open(target_path, 'w', encoding='utf-8') as f:

        random.shuffle(text_list)

        length = int(len(text_list) * sample)

        random_text = text_list[:length]

        for lines in tqdm(random_text):

            for line in lines:

                str = line[0] + ' ' + line[1] + '\n'

                f.write(str)

    print('文本切割成功，储存至{}，采样{}条数据'.format(target_path, length))


single_text_samples(train_sentences, data_train)
single_text_samples(dev_sentences, data_dev)
single_text_samples(test_sentences, data_test)