# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from data.data_util import num_samples, sour_lang, target_lang


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) # 构建embedding层
        self.gru = nn.GRU(hidden_size, hidden_size) # GRU单元

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) # embedding层，刚开始形状为【1， 神经元数量】，1代表当前送入的一个单词，这里固定为1
        output = embedded
        output, hidden = self.gru(output, hidden) # 当前单词进入GRU单元，输出形状为【1，1， 神经元数量】，隐状态【1，1， 神经元数量】
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length # 解码器中限制最大单词数

        self.embedding = nn.Embedding(self.output_size, self.hidden_size) # embedding层，参数为【367， 256】
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) # attention层，e此时的上一时刻的隐状态和此时刻的输入张量拼接，因为两个维度都是256，所以乘2， 解码器统一输出长度为10，因为编码器的输出句子长度是10，然后跟他点乘
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size) # attention机制层
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size) # GRU层，256 * 256
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) # 解码器的输入形状为【1， 1， 1 --》256】，
        embedded = self.dropout(embedded)
        # attn_weights 形状为【1， 10】，即输入句子的最大长度
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # 在attention机制，首先将此时的上一时刻的隐状态和此时刻的输入张量拼接
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), # batch matrix multiply
                                 encoder_outputs.unsqueeze(0)) # 新增一个维度，作为batch，这里attention与编码器输出开始矩阵相乘，【1，1，10】，编码器是【1，10,256】，batch相乘之后维度为【1，1,256】

        output = torch.cat((embedded[0], attn_applied[0]), 1) # 解码器的输入张量与attention层的拼接，因为都是256，所以这里是512个神经元
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output) # 【1，1，256】
        output, hidden = self.gru(output, hidden) # 两个都是【1，1，256】

        output = F.log_softmax(self.out(output[0]), dim=1) # 【1， 367】输出层，全部与映射到输出语句的单词数上
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
