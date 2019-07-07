#coding:utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# make_regression生成回归模型数据
from sklearn.datasets import make_regression

# 关键参数有n_samples（生成样本数），n_features（样本特征数），noise（样本随机噪音）和coef（是否返回回归系数
# X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征

samples = 1000
X, Y, coef = make_regression(n_samples=samples, n_features=1, noise=10, coef=True)

def sigmoid(x):

    return 1 / (1 + np.exp(-x))
X = np.array(X).reshape([-1, 1])
Y = sigmoid(np.array(Y).reshape([-1, 1]))
# plt.scatter(X, Y, c='b', s=3)
# plt.plot(X, Y, c='r')
# plt.xticks(())  # 不显示 x
# plt.yticks(())  # 不显示 y
# plt.show()

# print(x)
# print(y)

with tf.variable_scope('placeholder'):

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])
with tf.variable_scope('layer_1'):

    w = tf.Variable(tf.random_normal([1, 10]), dtype=tf.float32)
    b = tf.Variable(tf.random_normal([10]))

    h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b))

with tf.variable_scope('output'):

    w = tf.Variable(tf.random_normal([10, 1]), dtype=tf.float32)
    b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

    y_pred = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(h, w), b))
    y_true = tf.nn.sigmoid(y)

with tf.variable_scope('evalute'):

    lost = tf.reduce_mean(tf.square(y_pred -  y_true))
    opt = tf.train.AdamOptimizer(0.01).minimize(lost)

with tf.variable_scope('test'):

    acc = tf.reduce_mean((abs(y_pred - y_true) / y_true))

def train():

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        batch_size = 128
        epochs = 1000
        display_epoch = epochs // 10

        errors = []
        for epoch in range(epochs):

            batchs = samples // batch_size
            for i in range(batchs):
                x_batch, y_batch = X[i * batch_size: (i + 1 )* batch_size], Y[i * batch_size: (i + 1 )* batch_size]


                _ = sess.run(opt, feed_dict={x:x_batch, y:y_batch})

            error = sess.run(acc, feed_dict={x:x_batch, y:y_batch})
            errors.append(error)

        plt.plot(np.arange(epochs), errors)
        plt.show()
        #     if (epoch + 1) % display_epoch == 0:
        #
        #         cost = sess.run([lost], feed_dict={x:x_batch, y:y_batch})
        #
        #         print('epochs : {} | cost ：{}'.format(epoch, cost))
        #         print('y_ : {} | y_batch ：{}'.format(y_, y_batch))

        # saver = tf.train.Saver()
        # saver.save(sess, './tmp/tf练手')

        # 绘制图表

        error = sess.run(acc, feed_dict={x:x_batch, y:y_batch})

        print('精确率:',error)
        # print('预测值:', y_[0:5])
        # print('真实值:', sigmoid(y_batch[0:5]))
        # plt.scatter(y_, y_batch)
        # plt.show()

def test():

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('./tmp/tf练手.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./tmp'))
        sess.run(tf.global_variables_initializer())
        print(sess.run('output/Variable:0'))
# test()
train()