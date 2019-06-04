from collections import Counter

import numpy as np
from nltk.translate import bleu_score


def bp(references, candidate):
    # brevity penality,句子长度惩罚因子
    ind = np.argmin([abs(len(i) - len(candidate)) for i in references])
    if len(references[ind]) < len(candidate):
        return 1
    scale = 1 - (len(candidate) / len(references[ind]))
    return np.e ** scale


def parse_ngram(sentence, gram):
    # 把一个句子分成n-gram
    return [sentence[i:i + gram] for i in range(len(sentence) - gram + 1)]  # 此处一定要注意+1，否则会少一个gram


def sentence_bleu(references, candidate, weight):
    bp_value = bp(references, candidate)
    s = 1
    for gram, wei in enumerate(weight):
        gram = gram + 1
        # 拆分n-gram
        ref = [parse_ngram(i, gram) for i in references]
        can = parse_ngram(candidate, gram)
        # 统计n-gram出现次数
        ref_counter = [Counter(i) for i in ref]
        can_counter = Counter(can)
        # 统计每个词在references中的出现次数
        appear = sum(min(cnt, max(i.get(word, 0) for i in ref_counter)) for word, cnt in can_counter.items())
        score = appear / len(can)
        # 每个score的权值不一样
        s *= score ** wei
    s *= bp_value  # 最后的分数需要乘以惩罚因子
    return s


references = [
    "the dog jumps high",
    "the cat runs fast",
    "dog and cats are good friends"
]
candidate = "the d o g  jump s hig"
weights = [0.25, 0.25, 0.25, 0.25]
print(sentence_bleu(references, candidate, weights))
print(bleu_score.sentence_bleu(references, candidate, weights))