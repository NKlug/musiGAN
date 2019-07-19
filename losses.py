import tensorflow as tf


def softmax_x_entropy(logits, labels):
    labels = tf.cast(labels, dtype=tf.int32)
    if not isinstance(logits, list):
        logits = [logits]
    return tf.add_n(
        [tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=x, labels=tf.ones(shape=tf.shape(x)[:-1],
                                                                                        dtype=tf.int32) * labels)) for x
         in
         logits])
