import tensorflow as tf
import numpy as np

def gradient_loss(pred_patch, target_patch):
	loss_edge = tf.reduce_mean(tf.square(tf.abs(pred_patch[:,1:,1:,:] - pred_patch[:,:-1,1:,:]) - tf.abs(target_patch[:,1:,1:,:] - target_patch[:,:-1,1:,:])) 
		+ tf.square(tf.abs(pred_patch[:,1:,1:,:] - pred_patch[:,1:,:-1,:]) - tf.abs(target_patch[:,1:,1:,:] - target_patch[:,1:,:-1,:])))
	return(loss_edge)