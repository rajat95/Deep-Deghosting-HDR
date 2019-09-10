import tensorflow as tf
import cv2
import numpy as np
import os

def generate_reference_patches(img_seq_color, sigma_g=0.2, sigma_l=0.2, window_size=8):
    Burst_size, Batch_size, H, W, C = img_seq_color.shape # Burst x Batch x channel x H x W
    img_seq_color_batch = tf.reshape(img_seq_color, (Burst_size*Batch_size, H, W, C))
    mean_kr = np.ones((1, window_size, window_size, C))/(int(C)*(window_size**2))
    muY_seq = tf.layers.conv2d(img_seq_color_batch, 1, kernel_size=window_size, strides=1, kernel_initializer=tf.constant_initializer(mean_kr),trainable=False,name = 'muY_conv',reuse=tf.AUTO_REUSE)
    lY = tf.reduce_mean(img_seq_color_batch, axis=(1,2,3), keepdims=True) # Burst*Batch x 1 x 1 x 1
    muY_sq_seq = muY_seq*muY_seq # Burst*Batch x H' x W' x 1
    muY_seq_sq = tf.layers.conv2d(img_seq_color_batch*img_seq_color_batch,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(mean_kr),trainable=False,name = 'muY_sq_conv',reuse=tf.AUTO_REUSE) 
    sigma_sq_seq = muY_seq_sq - muY_sq_seq # Burst*Batch x H' x W'x 1
    sigmaY_sq_seq = tf.reshape(sigma_sq_seq, [Burst_size, Batch_size, sigma_sq_seq.shape[1], sigma_sq_seq.shape[2], 1]) # Burst x Batch x 1 x H' x W'
    sigmaY_sq = tf.reduce_max(sigmaY_sq_seq, axis=0)
    patch_index = tf.argmax(sigmaY_sq_seq, axis=0)
    denom_g = 2 * (sigma_g**2)
    denom_l = 2 * (sigma_l**2)
    LY = tf.exp((-1.0*((muY_seq-0.5)**2)/denom_g)+(-1.0*((lY-0.5)**2)/denom_l)) # Burst*Batch  x H' x W' x 1
    # print(torch.mean(LY))
    LY = tf.reshape(LY, [Burst_size, Batch_size, LY.shape[1], LY.shape[2], 1])
    lY = tf.reshape(lY, [Burst_size, Batch_size, 1, 1, 1])
    muY_seq = tf.reshape(muY_seq, [Burst_size, Batch_size, muY_seq.shape[1], muY_seq.shape[2], 1])
    muY = tf.reduce_sum(LY*muY_seq,axis=0)/tf.reduce_sum(LY,axis=0) # Batch x 1 x H' x W'
    muY_sq = muY*muY
    muY_sq_seq = tf.reshape(muY_sq_seq, [Burst_size,Batch_size,muY_sq_seq.shape[1],muY_sq_seq.shape[2], 1])
    return patch_index,muY,muY_sq,muY_seq,muY_sq_seq,sigmaY_sq,sigmaY_sq_seq


def mef_color(img_seq_color,fused_img, window_size=8, only_cost=True):
    C1 = 0.01**2
    C2 = 0.03**2
    params = generate_reference_patches(img_seq_color)
    patch_index, muY, muY_sq, muY_seq, muY_sq_seq, sigmaY_sq, sigmaY_sq_seq = params
    Burst_size, Batch_size, H, W, channels = img_seq_color.shape # Burst x Batch x H x W x channels
    print(channels)
    mean_kr = np.ones((1, window_size, window_size, channels))/(int(channels)*(window_size**2)) # 1  x 11 x 11 x 1
    print(mean_kr.dtype, 'mean_kr_dtype')
    muX = tf.layers.conv2d(fused_img, 1, kernel_size=8, strides=1, kernel_initializer=tf.constant_initializer(mean_kr), trainable=False, name = 'muX_conv', reuse=tf.AUTO_REUSE)
    muX_sq = muX*muX
    sq_muX = tf.layers.conv2d(fused_img*fused_img, 1, kernel_size=8, strides=1, kernel_initializer=tf.constant_initializer(mean_kr), trainable=False, name = 'sigmaX_conv', reuse=tf.AUTO_REUSE)
    sigmaX_sq = sq_muX - muX_sq # Batch x H' x W' x 1
    fused_img_expanded = tf.expand_dims(fused_img, 0)
    sig_xy_inter = fused_img_expanded*img_seq_color # Burst x Batch x H x W x channels
    sig_xy_inter = tf.reshape(sig_xy_inter, [Burst_size*Batch_size, H, W, channels])
#   sig_xy_inter = sig_xy_inter.contiguous().view(Burst_size*Batch_size,channels,H,W) # Burst*Batch x channel x H x W
    sig_xy = tf.layers.conv2d(sig_xy_inter, 1, kernel_size=8, strides=1, kernel_initializer=tf.constant_initializer(mean_kr), trainable=False, name = 'sigXY_conv', reuse=tf.AUTO_REUSE)
    sig_xy = tf.reshape(sig_xy, [Burst_size, Batch_size, muY_seq.shape[2], muY_seq.shape[3], 1])
    sigmaXY = sig_xy - (tf.expand_dims(muX,0)*muY_seq) # Burst x Batch x H' x W' x 1

    muX_exp = tf.expand_dims(muX, 0)
    muY_exp = tf.expand_dims(muY, 0)
    muX_sq_exp = tf.expand_dims(muX_sq, 0)
    muY_sq_exp = tf.expand_dims(muY_sq, 0)
    sigmaX_sq_exp = tf.expand_dims(sigmaX_sq, 0)
    qmap_int = ((2*muX_exp*muY_exp+C1)*(2*sigmaXY+C2))/((muX_sq_exp+muY_sq_exp+C1)*(sigmaX_sq_exp+sigmaY_sq_seq+C2)) # Burst x Batch  x H' x W' x 1
    print(patch_index.shape)
    print(qmap_int.shape, '_qmap_shape')
    index_one_hot = tf.one_hot(patch_index, depth = Burst_size, dtype=tf.float32, axis=0)
    print(index_one_hot.shape)
    print(qmap_int.shape, 'qmap_dtype')
    qmap = tf.reduce_sum(qmap_int*index_one_hot,axis=0)
    cost = tf.reduce_mean(qmap)
    if only_cost:
        return cost
    return cost,qmap_int, patch_index, qmap_int
