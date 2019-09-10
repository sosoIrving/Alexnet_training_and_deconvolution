import tensorflow as tf



def inference(images, dropout_rate=0.5, wd=None):                   #dropout_rate = 0.5
    with tf.compat.v1.variable_scope('conv1', reuse= tf.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1),
                                 trainable=True, name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(0.1, shape=[96], dtype=tf.float32), trainable=True,
                                 name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name='conv1')
    #lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool2d(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    with tf.compat.v1.variable_scope('conv2', reuse = tf.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=True, name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(0.1, shape=[256], dtype=tf.float32), trainable=True,
                                 name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='conv2')
    #lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool2d(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    with tf.compat.v1.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1),
                                 trainable=True, name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(0.1, shape=[384], dtype=tf.float32), trainable=True,
                                 name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name='conv3')

    with tf.compat.v1.variable_scope('conv4', reuse=tf.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1),
                                 trainable=True, name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(0.1, shape=[384], dtype=tf.float32), trainable=True,
                                 name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='conv4')

    with tf.compat.v1.variable_scope('conv5', reuse=tf.AUTO_REUSE):
        kernel = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=True, name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(0.1, shape=[256], dtype=tf.float32), trainable=True,
                                 name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name='conv5')
    pool5 = tf.nn.max_pool2d(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    flatten = tf.keras.layers.Flatten(name='flatten')(pool5)


    with tf.compat.v1.variable_scope('local1', reuse=tf.AUTO_REUSE):
        weights = tf.compat.v1.get_variable(
            initializer=tf.random.truncated_normal([6 * 6 * 256, 4096], dtype=tf.float32, stddev=1 / 4096.0), trainable=True,
            name='weights')
        if wd is not None:
            weights_loss = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.compat.v1.add_to_collection('losses', weights_loss)
        biases = tf.compat.v1.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True,
                                 name='biases')
        local1 = tf.nn.relu(tf.compat.v1.nn.xw_plus_b(flatten, weights, biases), name='local1')
        local1 = tf.nn.dropout(local1, dropout_rate)

    with tf.compat.v1.variable_scope('local2', reuse=tf.AUTO_REUSE):
        weights = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1 / 4096.0),
                                  trainable=True, name='weights')
        if wd is not None:
            weights_loss = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weights_loss)
        biases = tf.compat.v1.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True,
                                 name='biases')
        local2 = tf.nn.relu(tf.compat.v1.nn.xw_plus_b(local1, weights, biases), name='local2')
        local2 = tf.nn.dropout(local2, dropout_rate)

    with tf.compat.v1.variable_scope('local3', reuse=tf.AUTO_REUSE):
        weights = tf.compat.v1.get_variable(initializer=tf.random.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-3),
                                  trainable=True, name='weights')
        biases = tf.compat.v1.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=True,
                                 name='biases')
        local3 = tf.compat.v1.nn.xw_plus_b(local2, weights, biases, name='local3')

    return local3