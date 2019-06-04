from tflearn.datasets import oxflower17
import tensorflow as tf

def vgg11(x, n_class):

    layers = {

        # #################################################################
        # 权重
        # #################################################################
        # conv-1
        'wc1_1':tf.get_variable('wc1_1', [3, 3, 3, 64], tf.float32),

        # conv-2
        'wc2_1': tf.get_variable('wc2_1', [3, 3, 64, 128], tf.float32),

        # conv-3
        'wc3_1': tf.get_variable('wc3_1', [3, 3, 128, 256], tf.float32),
        'wc3_2': tf.get_variable('wc3_2', [3, 3, 256, 256], tf.float32),

        # conv-4
        'wc4_1': tf.get_variable('wc4_1', [3, 3, 256, 512], tf.float32),
        'wc4_2': tf.get_variable('wc4_2', [3, 3, 512, 512], tf.float32),

        # conv-5
        'wc5_1': tf.get_variable('wc5_1', [3, 3, 512, 512], tf.float32),
        'wc5_2': tf.get_variable('wc5_2', [3, 3, 512, 512], tf.float32),

        # fc
        'wfc1': tf.get_variable('wfc1', [7*7*512, 4096], tf.float32),
        'wfc2': tf.get_variable('wfc2', [4096, 4096], tf.float32),
        'wfc3': tf.get_variable('wfc3', [4096, n_class], tf.float32),
        # #################################################################
        # 偏置
        # #################################################################
        # conv-1
        'bc1_1': tf.get_variable('bc1_1', [64], tf.float32),

        # conv-2
        'bc2_1': tf.get_variable('bc2_1', [128], tf.float32),

        # conv-3
        'bc3_1': tf.get_variable('bc3_1', [256], tf.float32),
        'bc3_2': tf.get_variable('bc3_2', [256], tf.float32),

        # conv-4
        'bc4_1': tf.get_variable('bc4_1', [512], tf.float32),
        'bc4_2': tf.get_variable('bc4_2', [512], tf.float32),

        # conv-5
        'bc5_1': tf.get_variable('bc5_1', [512], tf.float32),
        'bc5_2': tf.get_variable('bc5_2', [512], tf.float32),

        # fc
        'bfc1': tf.get_variable('bfc1', [4096], tf.float32),
        'bfc2': tf.get_variable('bfc2', [4096], tf.float32),
        'bfc3': tf.get_variable('bfc3', [n_class], tf.float32),
    }

    # #################################################################
    # 卷积层 - 1
    # #################################################################
    net = tf.nn.conv2d(input=x, filter=layers['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc1_1']))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # #################################################################
    # 卷积层 - 2
    # #################################################################
    net = tf.nn.conv2d(input=net, filter=layers['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc2_1']))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # #################################################################
    # 卷积层 - 3
    # #################################################################
    net = tf.nn.conv2d(input=net, filter=layers['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc3_1']))
    net = tf.nn.conv2d(input=net, filter=layers['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc3_2']))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # #################################################################
    # 卷积层 - 4
    # #################################################################
    net = tf.nn.conv2d(input=net, filter=layers['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc4_1']))
    net = tf.nn.conv2d(input=net, filter=layers['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc4_2']))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # #################################################################
    # 卷积层 - 5
    # #################################################################
    net = tf.nn.conv2d(input=net, filter=layers['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc5_1']))
    net = tf.nn.conv2d(input=net, filter=layers['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, layers['bc5_2']))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # #################################################################
    # 全连接层
    # #################################################################
    net = tf.reshape(net, shape = [-1, layers['wfc1'].get_shape()[0]])
    net = tf.nn.relu(tf.matmul(net, layers['wfc1']) + layers['bfc1'])
    net = tf.nn.relu(tf.matmul(net, layers['wfc2']) + layers['bfc2'])
    net = tf.matmul(net, layers['wfc3']) + layers['bfc3']

    return net
import numpy as np
a = np.random.random([10, 224, 224, 3])
b = a[0:3]

if __name__ == '__main__':

    X, Y = oxflower17.load_data(one_hot = True)
    # X_samples = X[0:16, :, :, :]
    # Y_samples = Y[0:16, :]
    print(Y.shape)
    epochs = 100
    display_epoch = 10
    batch_size = 4
    n_class = Y.shape[1]
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y = tf.placeholder(tf.float32, shape=[None, n_class])
    y_pred = vgg11(x, n_class)
    lost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y)
    opt = tf.train.AdamOptimizer(0.1).minimize(lost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = X.shape[0] // batch_size
        for epoch in range(epochs):


            for i in range(total_batch):

                x_train = X[i * batch_size:(i + 1) * batch_size]
                y_train = Y[i * batch_size:(i + 1) * batch_size]
                sess.run(opt, feed_dict={x:x_train, y:y_train})

            if epoch % display_epoch == 0:

                cost = sess.run(lost, feed_dict={x:x_train, y:y_train})
                print('epoch:{}, lost:{}'.format(epoch, cost[0]))



