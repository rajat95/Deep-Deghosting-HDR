import numpy as np
import tensorflow as tf

def deepfuse_triple_untied(inp1, inp2, inp3):
    layers = tf.layers
    leaky_relu=tf.nn.leaky_relu
    with tf.variable_scope('DeepFuse'):
        prepooled_feats = [[],[],[]]
        num_downsample_layers = 4
        branch_enc = []
        num_feats = [32,64,128,256]
        for inp, name, idx in zip([inp1, inp2, inp3], ['inp1_', 'inp2_', 'inp3_'], [0,1,2]):
            conv = inp
            for i in range(num_downsample_layers):
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'enc_conv_'+name+str(i))
                prepooled_feats[idx].append(conv)
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'downsample_'+name+str(i))
            branch_enc.append(conv)

        
        merge=tf.concat([branch_enc[0],branch_enc[1], branch_enc[2]],axis=-1)
        conv = layers.conv2d(merge, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv1')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, prepooled_feats[0][-1],prepooled_feats[1][-1], prepooled_feats[2][-1]], axis=-1)
        conv = layers.conv2d(merge, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv2')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, prepooled_feats[0][-2],prepooled_feats[1][-2], prepooled_feats[2][-2]], axis=-1)

        conv = layers.conv2d(merge, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv3')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, prepooled_feats[0][-3],prepooled_feats[1][-3], prepooled_feats[2][-3]], axis=-1)

        conv = layers.conv2d(merge, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv4')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, prepooled_feats[0][-4],prepooled_feats[1][-4], prepooled_feats[2][-4]], axis=-1)

        # conv = layers.conv2d(merge, 8, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
        #                                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv5')
        conv = layers.conv2d(merge,3,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='fusion_output')
        return tf.clip_by_value(conv, 0.0, 1.0)

