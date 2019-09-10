import numpy as np
import tensorflow as tf
layers = tf.layers
leaky_relu=tf.nn.leaky_relu

def deepfuse_triple_tied(inp1, inp2, inp3):
    layers = tf.layers
    leaky_relu=tf.nn.leaky_relu
    with tf.variable_scope('DeepFuse'):
        prepooled_feats = [[],[],[]]
        num_downsample_layers = 4
        branch_enc = []
        num_feats = [32,64,128,256]
        for inp, reuse1, idx in zip([inp1, inp2, inp3], [None, True, True], [0,1,2]):
            conv = inp
            for i in range(num_downsample_layers):
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv'+str(i), reuse = reuse1)
                prepooled_feats[idx].append(conv)
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample'+str(i), reuse = reuse1)
            branch_enc.append(conv)

        pooled_feats1 = []
        pooled_feats2=[]
        for idx in range(num_downsample_layers):
            pooled_feats1.append(tf.maximum(tf.maximum(prepooled_feats[0][idx],prepooled_feats[1][idx]),prepooled_feats[2][idx]))
        merge1 = tf.maximum(tf.maximum(branch_enc[0],branch_enc[1]),branch_enc[2])

        for idx in range(num_downsample_layers):
            pooled_feats2.append((prepooled_feats[0][idx]+prepooled_feats[1][idx]+prepooled_feats[2][idx])/3.0)
        merge2 = (branch_enc[0]+branch_enc[1]+branch_enc[2])/3.0

        merge=tf.concat([merge1,merge2],axis=-1)
        conv = layers.conv2d(merge, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv1')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([pooled_feats1[-1],pooled_feats2[-1]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv2')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([pooled_feats1[-2],pooled_feats2[-2]],axis=-1)], axis=-1)

        conv = layers.conv2d(merge, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv3')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([pooled_feats1[-3],pooled_feats2[-3]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv4')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([pooled_feats1[-4],pooled_feats2[-4]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge,3,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='fusion_output')
        return tf.clip_by_value(conv, 0.0, 1.0)

def deepfuse_gen(inp):
    layers = tf.layers
    leaky_relu=tf.nn.leaky_relu
    with tf.variable_scope('DeepFuse', reuse=tf.AUTO_REUSE):
        neg_feats = []
        pos_feats = []
        num_downsample_layers = 4
        pos_branch_enc = []
        neg_branch_enc = []
        num_feats = [32,64,128,256]
        feats1 = layers.conv2d(inp, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                              kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv0')
        neg_feats.append(feats1)
        feats1 =  layers.conv2d(feats1, 32, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample0')
        feats2 = layers.conv2d(feats1, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv1')
        neg_feats.append(feats2)
        feats2 =  layers.conv2d(feats2, 64, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample1')
        feats3 = layers.conv2d(feats2, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv2')
        neg_feats.append(feats3)
        feats3 =  layers.conv2d(feats3, 128, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample2')
        feats4 = layers.conv2d(feats3, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv3')
        neg_feats.append(feats4)
        feats4 =  layers.conv2d(feats4, 256, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample3')

        neg_encoding = feats4


        for i in range(4):
            neg_feats[i] = [tf.reduce_max(neg_feats[i], axis=0, keepdims=True), tf.reduce_mean(neg_feats[i], axis=0, keepdims=True)]
       
        merged_encoding = tf.concat([tf.reduce_max(neg_encoding, axis=0, keepdims=True),
                                     tf.reduce_mean(neg_encoding, axis=0, keepdims=True)], axis=-1)
        conv = layers.conv2d(merged_encoding, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv1')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-1][0], neg_feats[-1][1]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv2')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-2][0], neg_feats[-2][1]],axis=-1)],axis=-1)

        conv = layers.conv2d(merge, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv3')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-3][0], neg_feats[-3][1]],axis=-1)],axis=-1)
        conv = layers.conv2d(merge, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv4')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-4][0], neg_feats[-4][1]],axis=-1)],axis=-1)
        # conv = layers.conv2d(merge, 8, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
        #                                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv5')
        conv = layers.conv2d(merge,3,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='fusion_output')
        return tf.clip_by_value(conv, 0.0, 1.0)

