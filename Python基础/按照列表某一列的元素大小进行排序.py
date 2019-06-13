from collections import Counter
import numpy as np
import codecs


with codecs.open("data/pos.txt", "r", encoding='utf-8') as f:
    pos_text = f.read()

# neg数据
with codecs.open("data/neg.txt", "r", encoding='utf-8') as f:
    neg_text = f.read()


total_text = pos_text + "\n" + neg_text
c = Counter(total_text.split())
d = sorted(c.most_common(), key=lambda x: x[1])
# c.most_common(5)
count = 0
for i in d:

    if i[1] == 1:
        print(i)
        count += 1


print(count)