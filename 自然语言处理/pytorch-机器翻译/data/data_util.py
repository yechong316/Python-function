import numpy as np
from io import open
import unicodedata
import string
import re
import random

import time
import math



'''
将文本进行缩小，提供
参数1，文本地址
参数2，切割比率，0-1之间，默认按0.1切割
返回一个新文本，文件名为原始文件名 + 语料条数
'''
num_samples = 5000
sour_lang = 'eng'
target_lang = 'fra'


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def fiter_eos(str):

    str = str.replace('<EOS>', '')
    str = str.replace('. ', '')

    return str


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def split_text(sour_lang, target_lang, text_path, num_samples=5000):


    with open(text_path, 'r', encoding='utf-8') as f:

        context = f.readlines()

        context = np.array(context)

        sour_num = len(context)
        context_index = context[np.random.permutation(sour_num)]


        new_context = context[:num_samples]


        new_text = '{}_{}_{}.txt'.format(sour_lang, target_lang, num_samples)
        with open(new_text, 'w+', encoding='utf-8') as new_f:

            for i in new_context:

                new_f.write(i)
    print('文本已成功切割，原始文本长度：{}， 切割文本长度：{}'.format(sour_num, num_samples))

    return new_text

if __name__ == '__main__':



    path = sour_lang + '-' + target_lang + '.txt'
    text_name = split_text(sour_lang, target_lang, path, num_samples=num_samples)