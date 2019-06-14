import math
from collections import OrderedDict, Counter
import os
import sys
import json


EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'



def load_dictionary(DICTIONARY_PATH):
    with open(with_path(DICTIONARY_PATH), 'r', encoding='UTF-8') as fp:
        dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
        index_word = OrderedDict()
        word_index = OrderedDict()
        for index, word in enumerate(dictionary):
            index_word[index] = word
            word_index[word] = index
        dim = len(dictionary)
    return dim, dictionary, index_word, word_index

def sentence_indice(sentence):
    ret = []
    for  word in sentence:
        if word in word_index:
            ret.append(word_index[word])
        else:
            ret.append(word_index[UNK])
    return ret

def indice_sentence(indice):
    ret = []
    for index in indice:
        word = index_word[index]
        if word == EOS:
            break
        if word != UNK and word != GO and word != PAD:
            ret.append(word)
    return ''.join(ret)

if __name__ == '__main__':

    DICTIONARY_PATH = r'D:\自然语言处理实战\应俊老师\Seq2Seq_Chatbot\Seq2Seq_Chatbot\db\dictionary.json'
    dim, dictionary, index_word, word_index = load_dictionary(DICTIONARY_PATH)
    print(dim, dictionary, index_word, word_index)