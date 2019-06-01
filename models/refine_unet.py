import tensorflow as tf

def leaky_relu(features, alpha = 0.2):
        return tf.nn.relu(features) - alpha * tf.nn.relu(-features)


def refine_net(input, name = 'occlusion_net', reuse=False):
    with tf.variable_scope(name):
        init = tf.contrib.keras.initializers.he_normal()
        conv1 = tf.layers.conv2d(input, 32, 3, 1, 'same', kernel_initializer=init, name='conv_1_1', reuse= reuse)
        conv1 = leaky_relu(conv1)
        conv1 = tf.layers.conv2d(conv1, 32, 3, 1, 'same', kernel_initializer=init, name='conv_1_2', reuse= reuse)
        conv1 = leaky_relu(conv1)
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name = 'pool1')

        conv2 = tf.layers.conv2d(pool1, 64, 3, 1, 'same', kernel_initializer=init, name='conv_2_1',reuse= reuse)
        conv2 = leaky_relu(conv2)
        conv2 = tf.layers.conv2d(conv2, 64, 3, 1, 'same', kernel_initializer=init, name='conv_2_2', reuse= reuse)
        conv2 = leaky_relu(conv2)
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')

        conv3 = tf.layers.conv2d(pool2, 128, 3, 1, 'same', kernel_initializer=init, name='conv_3_1', reuse= reuse)
        conv3 = leaky_relu(conv3)
        conv3 = tf.layers.conv2d(conv3, 128, 3, 1, 'same', kernel_initializer=init, name='conv_3_2', reuse= reuse)
        conv3 = leaky_relu(conv3)
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name = 'pool3')

        conv4 = tf.layers.conv2d(pool3, 256, 3, 1, 'same', kernel_initializer=init, name='conv_4_1', reuse= reuse)
        conv4 = leaky_relu(conv4)
        conv4 = tf.layers.conv2d(conv4, 256, 3, 1, 'same', kernel_initializer=init, name='conv_4_2', reuse= reuse)
        conv4 = leaky_relu(conv4)
        drop4 = conv4
        pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name = 'pool4')

        conv5 = tf.layers.conv2d(pool4, 512, 3, 1, 'same', kernel_initializer=init, name='conv_5_1', reuse= reuse)
        conv5 = leaky_relu(conv5)
        conv5 = tf.layers.conv2d(conv5, 512, 3, 1, 'same', kernel_initializer=init, name='conv_5_2', reuse= reuse)
        conv5 = leaky_relu(conv5)


        up6 = tf.contrib.keras.layers.UpSampling2D((2,2))(conv5)
        conv6 = tf.layers.conv2d(up6, 256, 2, 1, 'same',  kernel_initializer=init, name='conv_6_1', reuse= reuse)
        merge6 = tf.concat([conv4,conv6], axis = -1)
        conv6 = tf.layers.conv2d(merge6, 256, 3, 1, 'same',  kernel_initializer=init, name='conv_6_2', reuse= reuse)
        conv6 = leaky_relu(conv6)
        conv6 = tf.layers.conv2d(conv6, 256, 3, 1, 'same',  kernel_initializer=init, name='conv_6_3', reuse= reuse)
        conv6 = leaky_relu(conv6)

        up7 =  tf.contrib.keras.layers.UpSampling2D((2,2))(conv6)
        conv7 = tf.layers.conv2d(up7, 128, 2, 1, 'same',  kernel_initializer=init, name='conv_7_1', reuse= reuse)
        merge7 = tf.concat([conv3, conv7], axis = -1)
        conv7 = tf.layers.conv2d(merge7, 128, 3, 1, 'same',  kernel_initializer=init, name='conv_7_2', reuse= reuse)
        conv7 = leaky_relu(conv7)
        conv7 = tf.layers.conv2d(conv7, 128, 3, 1, 'same',  kernel_initializer=init, name='conv_7_3', reuse= reuse)
        conv7 = leaky_relu(conv7)

        up8 = tf.contrib.keras.layers.UpSampling2D((2,2))(conv7)
        conv8 = tf.layers.conv2d(up8, 64, 2, 1, 'same',  kernel_initializer=init, name='conv_8_1', reuse= reuse)
        merge8 = tf.concat([conv2, conv8], axis = -1)
        conv8 = tf.layers.conv2d(merge8, 64, 3, 1, 'same',  kernel_initializer=init, name='conv_8_2', reuse= reuse)
        conv8 = leaky_relu(conv8)
        conv8 = tf.layers.conv2d(conv8, 64, 3, 1, 'same',  kernel_initializer=init, name='conv_8_3', reuse= reuse)
        conv8 = leaky_relu(conv8)

        up9 =  tf.contrib.keras.layers.UpSampling2D((2,2))(conv8)
        conv9 = tf.layers.conv2d(up9, 32, 2, 1, 'same',  kernel_initializer=init, name='conv_9_1', reuse= reuse)
        merge9 = tf.concat([conv1, conv9], axis = -1)
        conv9 = tf.layers.conv2d(merge9, 32, 3, 1, 'same',  kernel_initializer=init, name='conv_9_2', reuse= reuse)
        conv9 = leaky_relu(conv9)
        conv9 = tf.layers.conv2d(conv9, 32, 3, 1, 'same',  kernel_initializer=init, name='conv_9_3', reuse= reuse)
        conv9 = leaky_relu(conv9)

        conv9 = tf.layers.conv2d(conv9, 3, 3, 1, 'same',  kernel_initializer=init, name='conv_9_4', reuse= reuse)
        conv9 = leaky_relu(conv9)

        conv10 = tf.layers.conv2d(conv9, 1, 3, 1, 'same',kernel_initializer=init, name='conv_10', reuse= reuse)
        output = tf.nn.sigmoid(conv10)
        return output

