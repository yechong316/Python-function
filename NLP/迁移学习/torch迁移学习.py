#coding: 'utf-8'

import torch
import torch.nn as nn
import copy
import math



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, mask=None, dropout=None):
    """因子化的点乘Attention-矩阵形式
    Query： 查询 （batch_size, heads, max_seq_len, d_k）
    Key: 键 (batch_size, heads, max_seq_len_d_k)
    Value: 值 （batch_size, heads, max_seq_len, d_v）
    d_v = d_k
    Q=K=V
    """
    d_k = query.size(-1)
    # (batch_size, heads, max_seq_len, d_k) * (batch_size, heads, d_k, max_seq_len)
    #  = (batch_size, heads, max_seq_len, max_seq_len)
    # 为了方便说明，只看矩阵的后两维 (max_seq_len, max_seq_len), 即
    #       How  are  you
    # How [[0.8, 0.2, 0.3]
    # are  [0.2, 0.9, 0.6]
    # you  [0.3, 0.6, 0.8]]
    # 矩阵中每个元素的含义是，他对其他单词的贡献（分数）
    # 例如，如果我们想得到所有单词对单词“How”的打分，取矩阵第一列[0.8, 0.2, 0.3], 然后做softmax
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # 对于padding部分，赋予一个极大的负数，softmax后该项的分数就接近0了，表示贡献很小
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 接着与Value做矩阵乘法:
    # (batch_size, heads, max_seq_len, max_seq_len) * (batch_size, heads, max_seq_len, d_k)
    # = (batch_size, heads, max_seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "heads is not a multiple of the number of the in_features"
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        这里的query, key, value与attention函数中的含义有所不同，这里指的是原始的输入.
        对于Encoder的自注意力来说，输入query=key=value=x
        对于Decoder的自注意力来说，输入query=key=value=t
        对于Encoder和Decoder之间的注意力来说， 输入query=t， key=value=m
        其中m为Encoder的输出，即给定target，通过key计算出m中每个输出对当前target的分数，在乘上m
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        ##   x: (batch_size, heads, max_seq_len, d_k)
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        ##   x: (batch_size, max_seq_len, d_k*h)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        ## output: (batch_size, max_seq_len, d_model)
        return self.linears[-1](x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2