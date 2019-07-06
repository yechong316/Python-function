#encoding=utf-8
from __future__ import print_function, unicode_literals
import sys
sys.path.append("../")
import jieba

import jieba.posseg as pseg

jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
jieba.del_word('自定义词')

# test_sent = ("北京大学生前来报到")
test_sent = ("北京大学报到")
words = jieba.cut(test_sent)
print('/'.join(words))
jieba.load_userdict("userdict.txt")
print('加载自定义字典后:')
words = jieba.cut(test_sent)
print('/'.join(words))




#
# result = pseg.cut(test_sent)
#
# for w in result:
#     print(w.word, "/", w.flag, ", ", end=' ')
#
# print("\n" + "="*40)
#
# terms = jieba.cut('easy_install is great')
# print('/'.join(terms))
# terms = jieba.cut('python 的正则表达式是好用的')
# print('/'.join(terms))
#
# print("="*40)
# # test frequency tune
# testlist = [
# ('今天天气不错', ('今天', '天气')),
# ('如果放到post中将出错。', ('中', '将')),
# ('我们中出了一个叛徒', ('中', '出')),
# ]
#
# for sent, seg in testlist:
#     print('/'.join(jieba.cut(sent, HMM=False)))
#     word = ''.join(seg)
#     print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
#     print('/'.join(jieba.cut(sent, HMM=False)))
# print("-"*40)
