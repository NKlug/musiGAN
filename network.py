import tensorflow as tf

from params import num_notes, magnitude_notes


def generator(input, feat_out):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x = tf.layers.flatten(input)
        x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu, name='dense1')
        x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu, name='dense2')

        x = tf.layers.dense(x, feat_out, activation=tf.nn.tanh, name='dense3')
        x = tf.reshape(x, (-1, num_notes, magnitude_notes))
        return x


def temporal_generator(input, feat_out):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x = tf.layers.flatten(input)
        x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu, name='dense1')
        x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu, name='dense2')

        x = tf.layers.dense(x, feat_out, activation=tf.nn.tanh, name='dense3')
        x = tf.reshape(x, (-1, num_notes, magnitude_notes + 1))
        return x


def discriminator(input):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = tf.layers.flatten(input)
        x = tf.layers.dense(x, 512, activation=tf.nn.leaky_relu, name='dense1')
        # x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense2')

        x = tf.layers.dense(x, 2, activation=None, name='dense3')

        return x
