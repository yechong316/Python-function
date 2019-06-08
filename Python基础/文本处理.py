import numpy as np

def load_text(text_path):
    '''
    导入文本，进行数据处理，返回正反向字典库，返回所有英文单词组成的列表
    :param text_path:
    :return:
    '''

    with open(text_path, 'r') as f:

        count = f.read()
        words = count.split()
        # print(words)
        vocb = np.unique(words)
        # print(vocb)
        # print(len(words))
        word2id, id2word = {}, {}
        id2word.update({
            i: vocb[i] for i in range(len(vocb))
        })
        word2id.update({
            vocb[i]:i for i in range(len(vocb))
        })
    # print(word2id)
    # print(len(id2word))


    return word2id, id2word, vocb, words