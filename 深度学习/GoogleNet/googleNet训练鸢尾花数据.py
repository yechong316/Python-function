from inception_v1 import *
from inception_v3 import *
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tflearn.datasets import oxflower17
def mess_dataset_order(train_data, train_label, dimention=2):

    '''
    将X,Y数据集乱序，并且返回，要求输入训练集的维度，默认label的维度为1

    :param train_data:
    :param train_label:
    :param dimention: 训练集的维度
    :return: 乱序后的训练集和label
    '''
    # assert dimention in [1, 2, 3, 4], '维度必须在1~4之间，否则请修改函数'
    assert type(dimention) == int
    permutation = np.random.permutation(train_label.shape[0])

    shuffled_labels = train_label[permutation]
    shuffled_dataset = train_data[permutation, :]
    if dimention == 3: shuffled_dataset = train_data[permutation, :, :]
    elif dimention == 4: shuffled_dataset = train_data[permutation, :, :, :]


    return shuffled_dataset, shuffled_labels

# ######################################################
# 定义模型参数
# ######################################################
tf.flags.DEFINE_string('path', r'../17flowers', '经典数据集地址')
tf.flags.DEFINE_float('learning_rate',0.002,'学习率')
tf.flags.DEFINE_float('dropout',1,'每层输出DROPOUT的大小')
tf.flags.DEFINE_integer('batch_size',32,'小批量梯度下降的批量大小')
tf.flags.DEFINE_float('sample', 0.1,'取样的数目')
tf.flags.DEFINE_integer('num_epoch',1000,'训练几轮')
tf.flags.DEFINE_integer('num_class',17,'一共多少类')
FLAGS = tf.flags.FLAGS


train_data, train_label = oxflower17.load_data(dirname=FLAGS.path ,
                            one_hot=True)

train_data, train_label = mess_dataset_order(train_data, train_label, dimention=train_data.shape[1])

sample = int(FLAGS.sample * train_data.shape[0])
x, y = train_data[:sample, :, :, :], train_label[:sample]


inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
output = tf.placeholder(tf.float32, [None, 17], name='output')
out, _ = inception_v3(inputs=inputs,
                 num_classes=FLAGS.num_class,
                 is_training=True,
                 dropout_keep_prob=FLAGS.dropout,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1')

lost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output, logits=out)

opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(lost)
######################
#evaluation
######################
acc_tf = tf.equal(tf.argmax(out, 1), tf.argmax(output, 1))
acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32), axis = None)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    batch_size = FLAGS.batch_size
    epochs = FLAGS.num_epoch
    for epoch in range(epochs):


        total_batch = x.shape[0] // batch_size  # 104
        for i in range(total_batch):
            X_train = x[i * batch_size: i * batch_size + batch_size]
            Y_train = y[i * batch_size: i * batch_size + batch_size]

            sess.run(opt, feed_dict={inputs: X_train, output: Y_train})


        if (epoch + 1) % 5 == 0:

                accuaray, cost = sess.run([acc, lost], feed_dict={inputs:X_train, output:Y_train})

                print('epoch = {} | cost = {} | acc = {}'.format(epoch + 1, cost[-1], accuaray))
            # y_pred = sess.run(out)
        learning_rate = 0.01 * (1 - epoch / epochs) ** 2
        # print('图片1的预测值为：', y_pred[0])
        # print('图片1的实际值为：', y[0])