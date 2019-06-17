import tensorflow as tf
import tqdm
import numpy as np
VOCAB_SIZE = 10384 # 10384
EMBEDDING_SIZE = 300

word_to_vec = {}
word_to_token = {}

# 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)

static_embeddings = static_embeddings.astype(np.float32)



# embeddings
with tf.name_scope("embeddings"):
    # 用pre-trained词向量来作为embedding层
    embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
    embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
    # 相加词向量得到句子向量
    sum_embed = tf.reduce_sum(embed, axis=1, name="sum_embed")