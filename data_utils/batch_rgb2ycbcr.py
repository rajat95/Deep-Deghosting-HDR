"""
Convert batches to RGB images to the Y-Cb-Cr color space.

"""


import numpy as np
import tensorflow as tf


def batch_rgb_to_ycbcr(inp, batch_size):
    rgb2yuv_filter = tf.constant(
                [[[[0.299, -0.169, 0.499],
                    [0.587, -0.331, -0.418],
                    [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    temp = tf.nn.conv2d(inp, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    return temp
