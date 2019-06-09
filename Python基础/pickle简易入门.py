import pickle
import tensorflow as tf

a = tf.random_normal(shape=[3, 4])

with tf.Session() as sess:

    data = sess.run(a)

with open('词向量.txt', 'wb') as f:

    print('dump:', data)
    pickle.dump(data, f, -1)

    print('successfully!')

with open('词向量.txt', 'rb') as f:

    new_data = pickle.load(f)
    print(new_data)