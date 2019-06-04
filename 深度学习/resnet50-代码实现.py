import tensorflow as tf
from tflearn.datasets import oxflower17#可以帮助我们下载所学数据集
from tensorflow.python.training import moving_averages
import numpy as np
import os
import time
from scipy.misc import imread, imresize

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d

# create weight variable
def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)

# conv2d layer
def conv2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        kernel = create_var("kernel", [kernel_size, kernel_size,
                                       num_inputs, num_outputs],
                            conv2d_initializer())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1],
                            padding="SAME")

# fully connected layer
def fc(x, num_outputs, scope="fc"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        weight = create_var("weight", [num_inputs, num_outputs],
                            fc_initializer())
        bias = create_var("bias", [num_outputs,],
                          tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, weight, bias)


# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = create_var("beta", [num_inputs,],
                               initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs,],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_var("moving_mean", [num_inputs,],
                                 initializer=tf.zeros_initializer(),
                                 trainable=False)
        moving_variance = create_var("moving_variance", [num_inputs],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


# avg pool layer
def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")

# max pool layer
def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                              [1, stride, stride, 1], padding="SAME")

class ResNet50:
    def __init__(self, inputs, num_classes=17, is_training=True,
                 scope="resnet50"):
        self.inputs =inputs
        self.is_training = is_training
        self.num_classes = num_classes

        with tf.variable_scope(scope):
            # construct the model
            net = conv2d(inputs, 64, 7, 2, scope="conv1") # -> [batch, 112, 112, 64]
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            net = max_pool(net, 3, 2, scope="maxpool1")  # -> [batch, 56, 56, 64]
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,
                              scope="block2")           # -> [batch, 56, 56, 256]
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")
                                                        # -> [batch, 28, 28, 512]
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")
                                                        # -> [batch, 14, 14, 1024]
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")
                                                        # -> [batch, 7, 7, 2048]
            net = avg_pool(net, 7, scope="avgpool5")    # -> [batch, 1, 1, 2048]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze") # -> [batch, 2048]
            self.logits = fc(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            self.predictions = tf.nn.softmax(self.logits)


    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride,
                                   is_training=is_training, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, is_training=is_training,
                                       scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = conv2d(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)
def load_datasets(folder_name):
    images, labels = [], []
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean = 0
    for i in range(len(class_name)):
        image_floder = folder_name + '/' + class_name[i]
        images_path = get_dir(image_floder)
        for j in images_path:
            image_o = np.expand_dims(imresize(imread(j, mode='RGB'), (224, 224)), axis=0)
            image = image_o - mean
            images.append(image)

            label = np.zeros(shape=len(class_name))
            label[i] = 1
            labels.append(np.expand_dims(label, axis=0))

    images_data = np.array(np.concatenate(images, axis=0))
    labels_data = np.array(np.concatenate(labels, axis=0))



    return images_data, labels_data
def access_classes(datasets_path):
    class_name = []
    file_path = []
    for root, dirs, files in os.walk(datasets_path):
        for dir in dirs:
            # print(dir)             #文件夹名
            class_name.append(dir)
    return class_name
if __name__ == "__main__":
    # folder_name = "./17flowers"
    # class_name = access_classes(folder_name)
    X, Y = oxflower17.load_data(one_hot=True)
    # print(X)
    print(X.shape)  # 1036,224,224,3
    # print(Y)
    print(Y.shape)  # 1036,17
    train_epoch = 1  # 训练迭代次数（训练数据遍历完一次，表示一次迭代）
    batch_size = 16  # 每个批次训练多少数据
    display_epoch = 100  # 每迭代100次，进行一次评估
    n_class = Y.shape[1]
    lr = tf.placeholder(tf.float32)  # 准备设计一个动态的学习率
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])  # 输入的数据
    y = tf.placeholder(tf.float32, [None, n_class])  # 标签数据

    # x = tf.random_normal([32, 224, 224, 3])
    resnet = ResNet50(x)
    pred=resnet.logits
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)  # 损失函数
    # loss = tf.nn.sigmoid_cross_entropy_with_logits
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # 优化器
    # one_hot 17位
    # [True,False,True....] X
    # True
    # 精度衡量定义
    acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32))
    # 4.训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化变量
        base_lr = 0.01
        learning_rate = base_lr
        for epoch in range(train_epoch):
            # batch_size=10,total_data=1360=>total_batch
            # 取数据
            total_batch = X.shape[0] // batch_size
            for i in range(total_batch):
                X_train, Y_train = X[i * batch_size:i * batch_size + batch_size], Y[
                                                                                  i * batch_size:i * batch_size + batch_size]
                sess.run(opt, feed_dict={x: X_train, y: Y_train, lr: learning_rate})
                # 进行评估
                cost, accuaray = sess.run([loss, acc], {x:X_train, y: Y_train, lr: learning_rate})
                print('step:%s,loss:%f,acc:%f' % (str(epoch) + '-' + str(i), cost[0], accuaray))
                # 动态修改学习率
                learning_rate = base_lr * (1 - epoch / train_epoch) ** 2
        save = tf.train.Saver()
        save.save(sess, './resnet/花')
