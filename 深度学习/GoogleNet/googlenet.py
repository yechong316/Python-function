import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 生成默认参数
def inception_v3_arg_scope(weight_decay=0.00004,                      # L2正则weight_decay
                           stddev=0.1,                                # 标准差
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon':0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # 提供了新的范围名称scope name
    # 对slim.conv2d和slim.fully_connected两个函数的参数自动赋值
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d], # 对卷积层的参数赋默认值
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev), # 权重初始化器
            activation_fn=tf.nn.relu,  # 激活函数用ReLU
            normalizer_params=batch_norm_params) as sc: # 标准化器参数用batch_norm_params
            return sc

# inputs为输入图片数据的tensor(299x299x3),scope为包含了函数默认参数的环境
def inception_v3_base(inputs, scope=None):
    # 保存某些关键节点
    end_points = {}
    # 定义InceptionV3的网络结构
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 设置卷积/最大池化/平均池化的默认步长为1,padding模式为VALID
        # 设置Inception模块组的默认参数
        with slim.arg_scope([slim.conv2d,       # 创建卷积层
                             slim.max_pool2d,   # 输出的通道数
                             slim.avg_pool2d],  # 卷积核尺寸
                            stride=1,           # 步长
                            padding='VALID'):   # padding模式
            # 经3个3x3的卷积层后,输入数据(299x299x3)变为(35x35x192),空间尺寸降低,输出通道增加
            net = slim.conv2d(inputs, 32, [3,3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')

            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')

        # 设置卷积/最大池化/平均池化的默认步长为1,padding模式为SAME
        # 步长为1,padding模式为SAME,所以图像尺寸不会变,仍为35x35
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            # 设置Inception Moduel名称为Mixed_5b
            with tf.variable_scope('Mixed_5b'):
                # 第1个分支：64输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：48输出通道的1x1卷积,连接64输出通道的5x5卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Con2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                # 第3个分支：64输出通道的1x1卷积,连接两个96输出通道的3x3卷积
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第4个分支：3x3的平均池化,连接32输出通道的1x1卷积
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                # 4个分支输出通道数之和＝64+64+96+32＝256,输出tensor为35x35x256
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第1个Inception模块组的第2个Inception Module
            with tf.variable_scope('Mixed_5c'):
                # 第1个分支：64输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：48输出通道的1x1卷积,连接64输出通道的5x5卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                # 第3个分支：64输出通道的1x1卷积,连接两个96输出通道的3x3卷积
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第4个分支：3x3的平均池化,连接64输出通道的1x1卷积
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸为35x35x288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第1个Inception模块组的第3个Inception Module
            with tf.variable_scope('Mixed_5d'):
                # 第1个分支：64输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：48输出通道的1x1卷积,连接64输出通道的5x5卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                # 第3个分支：64输出通道的1x1卷积,连接两个96输出通道的3x3卷积
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第4个分支：3x3的平均池化,连接64输出通道的1x1卷积
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸为35x35x288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第2个Inception模块组
            with tf.variable_scope('Mixed_6a'):
                # 第1个分支：3x3卷积,步长为2,padding模式为VALID,因此图像被压缩为17x17
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2 , padding='VALID', scope='Conv2d_1a_1x1')
                # 第2个分支：64输出通道的1x1卷积,连接2个96输出通道的3x3卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    # 步长为2,padding模式为VALID,因此图像被压缩为17x17
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                # 第3个分支：3x3的最大池化层,步长为2,padding模式为VALID,因此图像被压缩为17x17x256
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第2个Inception模块组,包含5个Inception Module
            with tf.variable_scope('Mixed_6b'):
                # 第1个分支：192输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：128输出通道的1x1卷积,接128输出通道的1x7卷积,接192输出通道的7x1卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸=17x17x(192+192+192+192)=17x17x768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 经过一个Inception Module输出tensor尺寸不变,但特征相当于被精炼类一遍
            # 第3个Inception模块组
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸为17x17x768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第4个Inception模块组
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸为17x17x768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第5个Inception模块组
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸为17x17x768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 将Mixed_6e存储于end_points中
            end_points['Mixed_6e'] = net
            # 第3个Inception模块
            # 第1个Inception模块组
            with tf.variable_scope('Mixed_7a'):
                # 第1个分支：192输出通道的1x1卷积,接320输出通道的3x3卷积 步长为2
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                # 第2个分支：4个卷积层
                with tf.variable_scope('Branch_1'):
                    # 192输出通道的1x1卷积
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    # 192输出通道的1x7卷积
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    # 192输出通道的7x1卷积
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    # 192输出通道的3x3卷积　步长为2,输出8x8x192
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 第3个分支：3x3的最大池化层,输出8x8x768
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                # 输出tensor尺寸：8x8x(320+192+768)=8x8x1280,尺寸缩小,通道数增加
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 第2个Inception模块组
            with tf.variable_scope('Mixed_7b'):
                # 第1个分支：320输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：384输出通道的1x1卷积
                # 分支内拆分为两个分支：384输出通道的1x3卷积+384输出通道的3x1卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                               slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                               slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                # 第3个分支：448输出通道的1x1卷积,接384输出通道的3x3卷积,分支内拆分为两个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    # 分支内拆分为两个分支：384输出通道的1x3卷积+384输出通道的3x1卷积
                    branch_2 = tf.concat([
                               slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                               slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                # 第4个分支：3x3的平均池化层,接192输出通道的1x1卷积,输出8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸：8x8x(320+768+768+192)=8x8x2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 第3个Inception模块组
            with tf.variable_scope('Mixed_7c'):
                # 第1个分支：320输出通道的1x1卷积
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第2个分支：384输出通道的1x1卷积
                # 分支内拆分为两个分支：384输出通道的1x3卷积+384输出通道的3x1卷积
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                               slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                               slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                # 第3个分支：448输出通道的1x1卷积,接384输出通道的3x3卷积,分支内拆分为两个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    # 分支内拆分为两个分支：384输出通道的1x3卷积+384输出通道的3x1卷积
                    branch_2 = tf.concat([
                               slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                               slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                # 第4个分支：3x3的平均池化层,接192输出通道的1x1卷积,输出8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 输出tensor尺寸：8x8x(320+768+768+192)=8x8x2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_points

# 全局平均池化
def inception_v3(inputs,
                 num_classes=1000,          # 最后分类数量
                 is_training=True,          # 是否是训练过程的标志
                 dropout_keep_prob=0.8,     # Dropout保留节点的比例
                 prediction_fn=slim.softmax,# 进行分类的函数
                 spatial_squeeze=True,      # 是否对输出进行squeeze操作,即去除维数为1的维度
                 reuse=None,                # tf.variable_scope的reuse默认值
                 scope='InceptionV3'):      # tf.variable_scope的scope默认值
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)
            # 设置卷积/最大池化/平均池化的默认步长为1,padding模式为SAME
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                # 辅助分类节点
                with tf.variable_scope('AuxLogits'):
                    # 5x5的平均池化,步长设为3,padding模式设为VALID
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                    aux_logits = slim.conv2d(aux_logits,768, [5, 5], weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_5x5')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                    if spatial_squeeze:
                        # 进行squeeze操作,去除维数为1的维度
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits
            # 处理正常的分类预测
            with tf.variable_scope('Logits'):
                # 8x8的平均池化层
                net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
                # Dropout层
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn= None, normalizer_fn=None,scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    # 进行squeeze操作,去除维数为1的维度
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            # 辅助节点
            end_points['Logits'] = logits
            # 利用Softmax对结果进行分类预测
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points



import math
from datetime import datetime
import time

# 评估每轮计算占用的时间
# 输入TensorFlow的Session,需要测评的算子target,测试的名称info_string
def time_tensorflow_run(session, target, info_string):
    # 定义预热轮数(忽略前10轮,不考虑显存加载等因素的影响)
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        # 持续时间
        duration = time.time()- start_time
        if i >= num_steps_burn_in:
            # 只考量10轮迭代之后的计算时间
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now().strftime('%X'), i - num_steps_burn_in, duration))
            # 记录总时间
            total_duration += duration
            total_duration_squared += duration * duration
    # 计算每轮迭代的平均耗时mn,和标准差sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    # 打印出每轮迭代耗时
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now().strftime('%X'), info_string, num_batches, mn, sd))


# Inception V3运行性能测试
if __name__ == '__main__':
    batch_size = 32
    height, width = 299, 299
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(inception_v3_arg_scope()):


        # 传入inputs获取logits,end_points
        logits, end_points = inception_v3(inputs, is_training=False)
    # 初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num_batches = 100
    # 测试Inception V3的forward性能
    time_tensorflow_run(sess, logits, 'Forward')