import numpy as np
import tensorflow as tf

n_samples = 1000
learning_rate = 0.01
batch_size = 100
n_steps = 5000

# Generate input data
x_data = np.arange(100, step=.1)
y_data = x_data + 20 * np.sin(x_data / 10)

# Reshape data
x_data = np.reshape(x_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

# Placeholders for batched input
x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# Do training
with tf.variable_scope('test'):
    w = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0))

    y_pred = tf.matmul(x, w) + b
    loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for _ in range(n_steps):
            indices = np.random.choice(n_samples, batch_size)
            x_batch = x_data[indices]
            y_batch = y_data[indices]
            _, loss_val = sess.run([opt, loss], feed_dict={x:x_batch, y:y_batch})

        print(w.eval())
        print(b.eval())
        print(loss_val)
    saver = tf.train.Saver()
    # Save the variables to disk.
    saver.save(sess, "/tmp/test.ckpt")
    # Restore variables from disk.
    saver.restore(sess, "/tmp/test.ckpt")
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'y': y_pred})})
    model_exporter.export(FLAGS.work_dir,
                          tf.constant(FLAGS.export_version),
                          sess)
