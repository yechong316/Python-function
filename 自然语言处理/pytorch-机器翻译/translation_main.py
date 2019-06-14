# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import sys
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from data.data_util import *
from model.seq2seq import *






from nltk.translate.bleu_score import sentence_bleu
MAX_LENGTH = 10

path = './data/{}_{}_{}.txt'.format(sour_lang, target_lang, num_samples)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def readLangs(lang1, lang2, path, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in tqdm(lines)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in tqdm(pairs)]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in tqdm(pairs) if filterPair(pair)]


def prepareData(lang1, lang2, path, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path, reverse) # 实例化传入的语种，统计其长度

    pairs = filterPairs(pairs) # 过滤过长，过短的句子，取出以i am开头的句子

    for pair in tqdm(pairs): # 统计单词数量，开始做词典 本次抽取900条句子

        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):

    try:

        sentence = [lang.word2index[word] for word in sentence.split(' ')]
    except:
        KeyError
        print('字典没有这个词哦')

        exit(0)
    else:
        return sentence


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)






def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden() # 编码器隐状态初始化为0，形状为【batch数，句子数，神经元数量】，这里不清楚为什么要多一个1维度

    encoder_optimizer.zero_grad() # 初始0梯度
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0) # 机器翻译没有做pad
    target_length = target_tensor.size(0)
    # 同样初始化编码器的输出，统一置位0，形状为【最大单词数， 神经元数量】
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    # 开始遍历输入的张量，当前句子，逐个单词送入神经网络，进入编码器的是输入张量，上一次的隐状态，第一次输入隐状态默认为0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden) # 输出当前单词的输出层和隐状态，注意，在for循环中，隐状态开始送入下一个单词中
        encoder_outputs[ei] = encoder_output[0, 0] # 编码器输出的形状为【1， 1， 256】，那么encoder_output【0,0】输出【256】将输出状态送入预先准备的列表中
    # 编码器的pad方式很巧妙，首先组建一个全0张量，【单词数，神经元数量】，然后跟输入单词的数量，依次填充，没有填充的自然是0
    decoder_input = torch.tensor([[SOS_token]], device=device) # 解码器刚开始的输入自然是'SOS'，start of sequences

    decoder_hidden = encoder_hidden # 将编码器的隐状态作为解码器第一次的输入隐状态，也就是所谓的中间向量C
    # 是否使用teacher_forcing，虽然用他可以提高精度，可能训练的参数不是很好，但是考虑到效率因素，我们用他，这里设定一个比率，办法很巧妙
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length): # 同样开始遍历解码器的输入句子，将输入语句，隐状态，编码器的输出送入解码器
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs) # 总结，解码器输出【1，367】，隐状态【1,1，256】，attention【1，10】
            topv, topi = decoder_output.topk(1) #'''沿给定dim维度返回输入张量input中 k 个最大值。如果不指定dim，则默认为input的最后一维。如果为largest为 False ，则返回最小的 k 个值。'''
            decoder_input = topi.squeeze().detach()  # detach from history as input, topi 形状为【1，267】，topv 【1，1】

            loss += criterion(decoder_output, target_tensor[di]) # 跟target求损失
            if decoder_input.item() == EOS_token: # 如果是终止符号，那么就停止
                break

    loss.backward()

    encoder_optimizer.step() # 运行优化器
    decoder_optimizer.step()

    return loss.item() / target_length # 求评价损失





def trainIters(encoder, decoder, n_iters, net, print_every=1, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # 采用SGD优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) # 根据迭代次数，选择出N组句子，每组句子随机抽取
                      for i in tqdm(range(n_iters))] # 并且在这个的地方通过嵌套3个函数，将每个句子中的每个单词根据早已经做好的正反向字典去查各自的ID号，
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1] # pair 顾名思义，语句对
        input_tensor = training_pair[0] # 取出输入语种，形状为 【当前句子的单词数， 1】，因为ID号肯定是一个嘛
        target_tensor = training_pair[1] # 取出输出语种，形状为 【当前句子的单词数， 1】
        # 将输入输入输出的int32张量，编码器解码器类属性，各自的优化器，损失准则，送入训练模块中
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


            save_path_enc = './result/spa_eng_enc'
            save_path_att = './result/spa_eng_att'

            torch.save(net[0].state_dict(), save_path_enc)
            torch.save(net[1].state_dict(), save_path_att)

    showPlot(plot_losses)



def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, pairs, n=10):
    # input_lang, output_lang, pairs = prepareData(sour_lang, target_lang, path, True)
    #
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)  # 输入语种的单词总数和编码器隐藏层单元数送入编码器
    #
    # PATH = './result/spa_eng_enc'
    # encoder1.load_state_dict(torch.load(PATH, map_location=device))
    #
    # PATH = './result/spa_eng_att'
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=drop_out, MAX_LENGTH=10).to(
    #     device)  # 将解码器单元数，输出语种的单词种类，dropout率送入待attention机制的解码器
    # attn_decoder1.load_state_dict(torch.load(PATH, map_location=device))
    scors = []

    print('开始计算BLEU指标，随机抽取{}个句子'.format(n))
    for i in tqdm(range(n)):
        pair = random.choice(pairs)
        output_words, _ = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        reference = fiter_eos(pair[0])
        candidate = fiter_eos(output_sentence)
        score = sentence_bleu(reference, candidate)
        scors.append(score)

    avg_score = np.mean(scors)
    print('随机{}个句子后，评价BLEU指标为{}'.format(n, avg_score))


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

def bleu():

    evaluateRandomly(encoder1, attn_decoder1, pairs)


def model_train():

    # 数去编码器，解码器，迭代次数，显示迭代次数，开始训练
    trainIters(encoder1, attn_decoder1, epochs, print_every=display, net=[encoder1, attn_decoder1])

def test():

    count = 1
    while count < 10:

        print('开始测试本系统...')
        a = sys.stdout.write("请输入：")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        output_words, _ = evaluate(encoder1, attn_decoder1, sentence)
        output_sentence = ' '.join(output_words)

        print('这是偶给您翻译的，看看哈~~~', output_sentence)


def work_mode(num):

    if num == 0: model_train()

    elif num == 1: bleu()

    elif num == 2: print('还请星星一下我哦~~~后续添加这个功能。')
        # test()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
hidden_size = 256
MAX_LENGTH = 10
drop_out = 0.1
epochs = 7000
display = 50
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
input_lang, output_lang, pairs = prepareData(sour_lang, target_lang, path, True)

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device) # 输入语种的单词总数和编码器隐藏层单元数送入编码器

PATH = './result/spa_eng_enc'
encoder1.load_state_dict(torch.load(PATH, map_location=device))

PATH = './result/spa_eng_att'
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=drop_out).to(device) # 将解码器单元数，输出语种的单词种类，dropout率送入待attention机制的解码器
attn_decoder1.load_state_dict(torch.load(PATH, map_location=device))


if __name__ == '__main__':

    '''
    0 : 训练模式
    1 : BLEU模式
    2 : 测试模式，注：测试模式暂不开放。
    '''
    work_mode(2)

