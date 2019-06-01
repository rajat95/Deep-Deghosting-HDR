import tensorflow as tf
import numpy as np
import os
import argparse
import sys
from termcolor import colored
import random
from termcolor import colored
import time
from tqdm import tqdm
import cv2

sys.path.append("./models")
sys.path.append("./data_utils")
sys.path.append("./losses")

from tf_warp import backward_warp
from PWCNet import pwc_net
from refine_unet import refine_net
from affine import Affine
from transform_utils import *
from skimage.transform import resize
from prepare_batch import data_pipeline_refine

parser = argparse.ArgumentParser()
parser.add_argument('--train_patch_list', default='./data_samples/refine_train.txt', help='link to list of training patches')
parser.add_argument('--val_patch_list', default='./data_samples/refine_test.txt', help='link to list of test patches')
parser.add_argument('--logdir', default='refine_maps/', help='path to training logs')
parser.add_argument('--iters', default=55001, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--image_dim', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--restore', type=int, default=0, help='Train over checkpoint of train afresh')
parser.add_argument('--restore_ckpt', default='refine_maps/checkpoints/model10.ckpt', help = 'which checkpoint to restore' )
parser.add_argument('--gpu', default=1, help = 'which gpu to use' )

# get forward and backward flow together
def network(im1, im2, im3, im12, im21, im23, fl1, fl2,  batch_size = 8, image_dim = 256, pool_type = 1,train=True):
  
    warped_im1 = backward_warp(im1, fl1)
    warped_im1 = tf.reshape(warped_im1, (batch_size, image_dim, image_dim, 3))
    refine_input = tf.concat([warped_im1, im21], axis=-1)
    output_map1 = refine_net(refine_input, name='dark_refine')

    warped_im3 = backward_warp(im3, fl2)
    warped_im3 = tf.reshape(warped_im3, (batch_size, image_dim, image_dim, 3))
    refine_input = tf.concat([warped_im3, im23], axis=-1)
    output_map3 = refine_net(refine_input, name='dark_refine', reuse=True)

    output_im1 = output_map1*im21+(1.-output_map1)*warped_im1
    output_im3 = output_map3*im23+(1.-output_map3)*warped_im3
    return output_im1, output_map1, warped_im1, output_im3, output_map3, warped_im3


def restore_weights(sess, ckpt_path = None):
    saver = tf.train.Saver(v for v in tf.all_variables() if 'pwc' not in v.name)
    saver.restore(sess, ckpt_path)


def train(batch_size = 8, image_dim = 256, logdir='./refine_maps', train_patch_list='./refine_train.txt', val_patch_list='./refine_test.txt',N_ITERS=55, restore = False, restore_ckpt = None):


    im1 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    im2 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    im3 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    im12 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    im21 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    im23 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    flow21 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 2])
    flow23 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 2])
    gt = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    gt1 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    gt2 = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, 3])
    regularizer_coeff = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    output1, output_map1, warped_im1, output3, output_map3, warped_im3  = network(im1, im2,im3, im12, im21,im23,flow21,flow23)
    loss = tf.losses.mean_squared_error(output1, gt1)+tf.losses.mean_squared_error(output3, gt2)
    regularizer = tf.reduce_mean((output_map1**2.0))+ tf.reduce_mean((output_map3**2.0))
    overall_loss = loss+regularizer_coeff*regularizer
    print('total trainable vars :' + str(len(tf.trainable_variables())))
    pwc_vars = [v for v in tf.trainable_variables() if 'pwcnet' in v.name]
    print(str(len(pwc_vars)) + ' variables found in pwc_net')
    fuse_vars = [v for v in tf.trainable_variables() if 'DeepFuse' in v.name]
    print(str(len(fuse_vars)) + ' variables found in rest of the net')
    trainable_vars = [v for v in tf.trainable_variables() if v not in pwc_vars + fuse_vars]
    print('total trainable vars for refinement are ', len(trainable_vars))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    loss = tf.summary.scalar('overall_loss', overall_loss)
   
    train_step = tf.train.AdamOptimizer(lr).minimize(overall_loss, var_list=trainable_vars, global_step=global_step)
     
    inp1_im = tf.summary.image('training_image1', tf.cast(im1*255.0, tf.uint8))
    inp2_im = tf.summary.image('training_image2', tf.cast(im2*255.0, tf.uint8))
    inp3_im = tf.summary.image('training_image3', tf.cast(im3*255.0, tf.uint8))

    gt1_im =  tf.summary.image('training_image_gt1', tf.cast(gt1*255.0, tf.uint8))
    gt2_im =  tf.summary.image('training_image_gt2', tf.cast(gt2*255.0, tf.uint8))


    output_im = tf.summary.image('train_output_image1', output1)
    weight_map = tf.summary.image('train_weight_map1', output_map1)
    warp_im = tf.summary.image('train_warped_im1', warped_im1)


    output_im3 = tf.summary.image('train_output_image3', output3)
    weight_map3 = tf.summary.image('train_weight_map3', output_map3)
    warp_im3 = tf.summary.image('train_warped_im3', warped_im3)

    
    image_summary = tf.summary.merge([inp1_im, inp2_im, inp3_im, output_im, weight_map, warp_im,
                                      output_im3, weight_map3, warp_im3,gt1_im, gt2_im])
    loss_summary =  tf.summary.merge([loss])

    val_inp1_im = tf.summary.image('val_image1', im1)
    val_inp2_im = tf.summary.image('val_image2', im2)
    val_inp3_im = tf.summary.image('val_image3', im3)
    val_gt_im1 = tf.summary.image('val_gt1', gt1)
    val_gt_im2 = tf.summary.image('val_gt2', gt2)

    val_output_im1 = tf.summary.image('val_output_image1', output1)
    val_weight_map1 = tf.summary.image('val_weight_map1', output_map1)
    val_warped_im1 = tf.summary.image('val_warped_im1', warped_im1)

    val_output_im3 = tf.summary.image('val_output_image3', output3)
    val_weight_map3 = tf.summary.image('val_weight_map3', output_map3)
    val_warped_im3 = tf.summary.image('val_warped_im3', warped_im3)

    val_image_summary = tf.summary.merge([val_inp1_im, val_inp2_im, val_inp3_im, val_output_im1, val_weight_map1, 
                                          val_warped_im1, val_output_im3, val_weight_map3,val_warped_im3,
                                          val_gt_im1, val_gt_im2])
    

    val_mse = tf.losses.mean_squared_error(output1, gt1) + tf.losses.mean_squared_error(output3, gt2)
    val_summary = tf.summary.scalar('val_mse_tag', val_mse)
    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            restore_weights(sess, restore_ckpt)

        train_graph_writer = tf.summary.FileWriter(logdir+'/train/', sess.graph)
        val_graph_writer = tf.summary.FileWriter(logdir+'/val/', sess.graph)

        train_dataset = data_pipeline_refine(train_patch_list, batch_size, image_dim)
        train_iterator = train_dataset.make_initializable_iterator()
        train_batch = train_iterator.get_next()

        val_dataset = data_pipeline_refine(val_patch_list, batch_size, image_dim)
        val_iterator = val_dataset.make_initializable_iterator()
        val_batch = val_iterator.get_next()
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)

        for itr in tqdm(range(N_ITERS)):
            if itr<10000:
                reg_c = 3e-3
                lr_val = 1e-4
            elif itr<15000:
                reg_c = 2.5e-5
                lr_val = 0.75e-4
            elif itr<20000:
                reg_c = 2.0e-5
                lr_val = 0.5e-4
            elif itr<25000:
                reg_c = 1.75e-5
                lr_val = 0.25e-4
            elif itr<30000:
                reg_c = 1.5e-5
                lr_val = 0.25e-4
            
            img1, img2, img3, img12, img21, img23, ground_truth1, ground_truth2, flow1, flow2 = sess.run(train_batch)
            gs, loss_write, image_write = sess.run([train_step, loss_summary, image_summary],
                                                   feed_dict={im1: img1, im2: img2,im3:img3, im12:img12, im21:img21, im23:img23,
                                                              gt1: ground_truth1, gt2: ground_truth2, regularizer_coeff:reg_c,
                                                              flow21:flow1, flow23:flow2, lr:lr_val})
            train_graph_writer.add_summary(loss_write, global_step.eval())
            if (itr % 30 == 0):
                train_graph_writer.add_summary(image_write, global_step.eval())
            saver.save(sess, logdir + '/checkpoints/' + 'model' + str(itr) + '.ckpt')
            if itr%5000 == 0:
                total_val_mse = 0
                total_loss = 0
         
                for val_batch_index in tqdm(range(0, len(val_patch_list) // batch_size)):
                    img1, img2, img3, img12, img21,img23, ground_truth1, ground_truth2,flow1, flow2 = sess.run(val_batch)
                    mse_value, val_summ, val_image_write,loss = sess.run(
                                                                [val_mse, val_summary, val_image_summary, overall_loss],
                                                                feed_dict={im1: img1, im2: img2,im3:img3, im12:img12, im21:img21, im23:img23,
                                                                           gt1: ground_truth1, gt2: ground_truth2, regularizer_coeff:reg_c,
                                                                           flow21:flow1, flow23:flow2})

                    total_val_mse += float(mse_value)
                    total_loss += float(loss)

                    val_graph_writer.add_summary(val_summ, global_step.eval() + val_batch_index)
                    if (val_batch_index % 30 == 0):
                        val_graph_writer.add_summary(val_image_write, global_step.eval())
#            total_val_mse /= (len(val_patch_list) // batch_size)
                print(colored('mse: '+str(total_val_mse)+'loss: '+str(total_loss)))
        # val_fileLog.write(str(total_val_mse) + '\n')
        train_graph_writer.close()
        val_graph_writer.close()

 
if __name__=='__main__':
    opts = parser.parse_args()
    train_idx_file = opts.train_patch_list
    with open(train_idx_file, 'r') as f:
        train_patches = eval(f.readline())
    val_idx_file = opts.val_patch_list
    with open(val_idx_file, 'r') as f:
        val_patches = eval(f.readline())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)

    print(val_patches)
    train(batch_size = opts.batch_size,
          image_dim = opts.image_dim,
          logdir = opts.logdir,
          N_ITERS = opts.iters,
          restore = opts.restore,
          restore_ckpt = opts.restore_ckpt,
          train_patch_list = train_patches, 
          val_patch_list=val_patches)
