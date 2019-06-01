import numpy as np
import tensorflow as tf
from scipy.misc import imread

def calc_histogram(image):
    values_range = tf.constant([0, 255], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
    return histogram


def histogram_loss(img1, img2):
	hist1 = calc_histogram(img1)
	hist2 = calc_histogram(img2)
	loss = tf.losses.mean_squared_error(hist1, hist2)
	return loss

def batch_hist_loss(batch1, batch2, batch_size):
	batch_hist_val = 0
	for batch_index in range(batch_size):
		batch_hist_val += histogram_loss(batch1[batch_index], batch2[batch_index])
	return(batch_hist_val / batch_size)
