import numpy as np
import tensorflow as tf
layers = tf.layers
leaky_relu=tf.nn.leaky_relu

def deepfuse_triple_3encoder(inp1, inp2, inp3):
    layers = tf.layers
    leaky_relu=tf.nn.leaky_relu
    with tf.variable_scope('DeepFuse'):
        neg_prepooled_feats = [[],[]]
        pos_prepooled_feats = [[],[]]
        num_downsample_layers = 4
        pos_branch_enc = []
        neg_branch_enc = []
        global_feats_pos = []
        global_feats_neg = []
        ref_feats = []
        num_feats = [32,64,128,256]
        for inp, reuse1, idx in zip([inp1, inp2], [None, True], [0,1]):
            conv = inp
            for i in range(num_downsample_layers):
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_enc_conv'+str(i), reuse = reuse1)
                neg_prepooled_feats[idx].append(conv)
                if inp == inp2:
                    global_feats_neg.append(conv)
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_downsample'+str(i), reuse = reuse1)

            neg_branch_enc.append(conv)

        for inp, reuse1, idx in zip([inp2, inp3], [None, True], [0,1]):
            conv = inp
            for i in range(num_downsample_layers):
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_enc_conv'+str(i), reuse = reuse1)
                pos_prepooled_feats[idx].append(conv)
                if inp == inp2:
                    global_feats_pos.append(conv)
                conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_downsample'+str(i), reuse = reuse1)

            pos_branch_enc.append(conv)

        conv = inp2
        for i in range(num_downsample_layers):
            conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_enc_conv'+str(i))
            ref_feats.append(conv)
            conv = layers.conv2d(conv, num_feats[i], kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_downsample'+str(i))

        ref_branch_enc = conv


        pos_pooled_feats1 = []
        neg_pooled_feats1 = []

        pos_pooled_feats2 = []
        neg_pooled_feats2 = []

        for idx in range(num_downsample_layers):
            neg_pooled_feats1.append(tf.maximum(neg_prepooled_feats[0][idx],neg_prepooled_feats[1][idx]))
            neg_merge1 = tf.maximum(neg_branch_enc[0],neg_branch_enc[1])
            pos_pooled_feats1.append(tf.maximum(pos_prepooled_feats[0][idx],pos_prepooled_feats[1][idx]))
            pos_merge1 = tf.maximum(pos_branch_enc[0],pos_branch_enc[1])

        for idx in range(num_downsample_layers):
            neg_pooled_feats2.append((neg_prepooled_feats[0][idx]+neg_prepooled_feats[1][idx])/2.0)
            neg_merge2 = (neg_branch_enc[0]+neg_branch_enc[1])/2.0
            pos_pooled_feats2.append((pos_prepooled_feats[0][idx]+pos_prepooled_feats[1][idx])/2.0)
            pos_merge2 = (pos_branch_enc[0]+pos_branch_enc[1])/2.0

        merge=tf.concat([neg_merge1,neg_merge2, pos_merge1, pos_merge2, ref_branch_enc],axis=-1)

        conv = layers.conv2d(merge, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv1')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_pooled_feats1[-1],neg_pooled_feats2[-1], pos_pooled_feats1[-1], pos_pooled_feats2[-1], ref_feats[-1]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv2')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_pooled_feats1[-2],neg_pooled_feats2[-2],pos_pooled_feats1[-2],pos_pooled_feats2[-2], ref_feats[-2]],axis=-1)], axis=-1)

        conv = layers.conv2d(merge, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv3')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_pooled_feats1[-3],neg_pooled_feats2[-3], pos_pooled_feats1[-3],pos_pooled_feats2[-3], ref_feats[-3]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv4')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_pooled_feats1[-4],neg_pooled_feats2[-4], pos_pooled_feats1[-4], pos_pooled_feats2[-4], ref_feats[-4]],axis=-1)], axis=-1)

        conv = layers.conv2d(merge,3,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='fusion_output')
        return tf.clip_by_value(conv, 0.0, 1.0)


#Assume all images in the batch belong to same sequence; to be only used for validation; can have different lengths positive and negative sequence
def deepfuse_gen_3encoder(neg_inp, pos_inp, ref):
    layers = tf.layers
    leaky_relu=tf.nn.leaky_relu
    with tf.variable_scope('DeepFuse', reuse=tf.AUTO_REUSE):
        neg_feats = []
        pos_feats = []
        ref_feats = []
        num_downsample_layers = 4
        pos_branch_enc = []
        neg_branch_enc = []
        num_feats = [32,64,128,256]
        feats1 = layers.conv2d(neg_inp, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                              kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_enc_conv0')
        neg_feats.append(feats1)
        feats1 =  layers.conv2d(feats1, 32, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_downsample0')
        feats2 = layers.conv2d(feats1, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_enc_conv1')
        neg_feats.append(feats2)
        feats2 =  layers.conv2d(feats2, 64, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_downsample1')
        feats3 = layers.conv2d(feats2, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_enc_conv2')
        neg_feats.append(feats3)
        feats3 =  layers.conv2d(feats3, 128, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_downsample2')
        feats4 = layers.conv2d(feats3, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_enc_conv3')
        neg_feats.append(feats4)
        feats4 =  layers.conv2d(feats4, 256, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'neg_downsample3')

        neg_encoding = feats4

        feats1 = layers.conv2d(pos_inp, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                              kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_enc_conv0')
        pos_feats.append(feats1)
        feats1 =  layers.conv2d(feats1, 32, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_downsample0')
        feats2 = layers.conv2d(feats1, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_enc_conv1')
        pos_feats.append(feats2)
        feats2 =  layers.conv2d(feats2, 64, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_downsample1')
        feats3 = layers.conv2d(feats2, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_enc_conv2')
        pos_feats.append(feats3)
        feats3 =  layers.conv2d(feats3, 128, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_downsample2')
        feats4 = layers.conv2d(feats3, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_enc_conv3')
        pos_feats.append(feats4)
        feats4 =  layers.conv2d(feats4, 256, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'pos_downsample3')
         
        pos_encoding = feats4


        feats1 = layers.conv2d(ref, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                              kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_enc_conv0')
        ref_feats.append(feats1)
        feats1 =  layers.conv2d(feats1, 32, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_downsample0')
        feats2 = layers.conv2d(feats1, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_enc_conv1')
        ref_feats.append(feats2)
        feats2 =  layers.conv2d(feats2, 64, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_downsample1')
        feats3 = layers.conv2d(feats2, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_enc_conv2')
        ref_feats.append(feats3)
        feats3 =  layers.conv2d(feats3, 128, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_downsample2')
        feats4 = layers.conv2d(feats3, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                               kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_enc_conv3')
        ref_feats.append(feats4)
        feats4 =  layers.conv2d(feats4, 256, kernel_size = 3, strides = 2, padding = 'SAME',activation = leaky_relu,
                                kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'ref_downsample3')
         
        ref_encoding = feats4

        for i in range(4):
            neg_feats[i] = [tf.reduce_max(neg_feats[i], axis=0, keepdims=True), tf.reduce_mean(neg_feats[i], axis=0, keepdims=True)]
            pos_feats[i] = [tf.reduce_max(pos_feats[i], axis=0, keepdims=True), tf.reduce_mean(pos_feats[i], axis=0, keepdims=True)]
       
        merged_encoding = tf.concat([tf.reduce_max(neg_encoding, axis=0, keepdims=True),
                                     tf.reduce_mean(neg_encoding, axis=0, keepdims=True),
                                     tf.reduce_max(pos_encoding, axis=0, keepdims=True),
                                     tf.reduce_mean(pos_encoding, axis=0, keepdims=True), 
                                     ref_encoding], axis=-1)
        conv = layers.conv2d(merged_encoding, 256, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv1')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-1][0], neg_feats[-1][1], pos_feats[-1][0], pos_feats[-1][1], ref_feats[-1]],axis=-1)], axis=-1)
        conv = layers.conv2d(merge, 128, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv2')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-2][0], neg_feats[-2][1], pos_feats[-2][0], pos_feats[-2][1], ref_feats[-2]],axis=-1)],axis=-1)

        conv = layers.conv2d(merge, 64, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv3')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-3][0], neg_feats[-3][1], pos_feats[-3][0], pos_feats[-3][1], ref_feats[-3]],axis=-1)],axis=-1)
        conv = layers.conv2d(merge, 32, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
                                                            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv4')
        conv = tf.keras.layers.UpSampling2D((2,2))(conv)
        merge = tf.concat([conv, tf.concat([neg_feats[-4][0], neg_feats[-4][1], pos_feats[-4][0], pos_feats[-4][1], ref_feats[-4]],axis=-1)],axis=-1)
        # conv = layers.conv2d(merge, 8, kernel_size = 3, strides = 1, padding = 'SAME', activation = leaky_relu,
        #                                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), name = 'dec_conv5')
        conv = layers.conv2d(merge,3,kernel_size=3,strides=1,padding='SAME',activation=tf.nn.sigmoid,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name='fusion_output')
        return tf.clip_by_value(conv, 0.0, 1.0)

