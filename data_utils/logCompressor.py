import tensorflow as tf


def log_compressor(im, MU = 5000.):
    return tf.log(1+MU*im)/tf.log(1+MU)