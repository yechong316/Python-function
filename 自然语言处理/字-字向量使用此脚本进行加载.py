import codecs
import numpy as np
from tqdm import tqdm, trange


def load_char_ver(emb_path, word_dim):
    '''

    :param emb_path: 字向量路径
    :param word_dim:  字向量维度
    :return: 形如{字：字向量}的字典
    '''
    pre_trained = {}  # key：字，eg，中等等，values：对应的词向量
    emb_invalid = 0
    for i, line in enumerate(tqdm(codecs.open(emb_path, 'r', 'utf-8'))):

        '如果第一行是字向量的个数以及维度，那么把下面的i == 0 改为 i == -1，不对第一行进行过滤'
        if i == 0:

            continue
        else:

            # 删除末尾的换行符
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:

                # 统计不符合预设维度的个数
                emb_invalid += 1

if __name__ == "__main__":
    word_dim = 300
    emb_path = r"D:\迅雷下载\token_vec_300.bin"

    pre_trained = load_char_ver(emb_path, word_dim)