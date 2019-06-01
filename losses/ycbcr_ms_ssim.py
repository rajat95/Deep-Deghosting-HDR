import tensorflow as tf
import numpy as np

def batch_rgb_to_ycbcr(inp, batch_size):
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
            [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    temp = tf.nn.conv2d(inp, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # print(temp.shape)
    return temp

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    # print(img1.shape, img2.shape)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value) + 1e-8
    return value


def tf_ms_ssim(img1, img2, batch_size, mean_metric=True, level=5):
    with tf.variable_scope("ms_ssim_loss"):
        img1 = tf.expand_dims(batch_rgb_to_ycbcr(img1, batch_size)[:, :, :, 0], -1)
        img2 = tf.expand_dims(batch_rgb_to_ycbcr(img2, batch_size)[:, :, :, 0], -1)
        # print(img1.shape, img2.shape)
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value



def tf_color_ssim(img1, img2, win_size = 8, cs_map=True, mean_metric=True):

    window = np.ones((win_size,win_size), dtype=np.float32)/(win_size*win_size*3)
    # print(img1.shape, img2.shape)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    conv_out11 = tf.layers.conv2d(img1,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(window),trainable=False,name = 'conv1_im1',reuse=tf.AUTO_REUSE)
    mu1 = tf.squeeze(conv_out11,[3])
    mu1_sq = mu1*mu1

    conv_out12 = tf.layers.conv2d(img2,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(window),trainable=False,name = 'conv1_im2',reuse=tf.AUTO_REUSE)
    mu2 = tf.squeeze(conv_out12,[3])
    mu2_sq = mu2*mu2

    mu1_mu2 = mu1*mu2


    conv_out21 = tf.layers.conv2d(img1*img1,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(window),trainable=False,name = 'conv2_im1',reuse=tf.AUTO_REUSE)
    sigma1_sq = tf.squeeze(conv_out21,[3]) - mu1_sq


    conv_out22 = tf.layers.conv2d(img2*img2,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(window),trainable=False,name = 'conv2_im2',reuse=tf.AUTO_REUSE)
    sigma2_sq = tf.squeeze(conv_out22,[3]) - mu2_sq

    conv_out212 = tf.layers.conv2d(img1*img2,1,kernel_size=8,strides=1,kernel_initializer=tf.constant_initializer(window),trainable=False,name = 'conv2_im12',reuse=tf.AUTO_REUSE)
    sigma12 = tf.squeeze(conv_out212,[3]) - mu1_mu2

    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value) + 1e-8
    return value

def tf_color_ms_ssim(img1, img2, batch_size, mean_metric=True, level=5):
    with tf.variable_scope("ms_ssim_loss"):
        #img1 = tf.expand_dims(batch_rgb_to_ycbcr(img1, batch_size)[:, :, :, 0], -1)
        #img2 = tf.expand_dims(batch_rgb_to_ycbcr(img2, batch_size)[:, :, :, 0], -1)
        # print(img1.shape, img2.shape)
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = tf_color_ssim(img1, img2, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean((ssim_map+1.0)/2.0))
            mcs.append(tf.reduce_mean((cs_map+1.0)/2.0))
            filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


