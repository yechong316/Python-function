# 前言：

本项目是基于Pytorch的官方教程（链接：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 基础上进行开发，后续会新增多国语言的调试任务。

# 概述

因个人电脑因素，本次项目随机抽样5000个样本进行分析。

# 使用方法

run --- translation_main.py即可。

看下文即可。

```
'''
0 : 训练模式
1 : BLEU模式
2 : 测试模式，注：测试模式暂不开放。
'''
work_mode(2)

***************分割线***************
def work_mode(num):

    if num == 0: model_train()

    elif num == 1: bleu()

    elif num == 2: print('还请星星一下我哦~~~后续添加这个功能。')
```

# 最终结果

随机抽取100个句子，BLEU指标在为0.78。原因暂时定位语料库句子太少。

# 计算时间

在GTX1060， 6G显卡容量，迭代次数为7000，计算时间大概在15min分钟。





