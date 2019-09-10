import tensorflow as tf

import argparse
import os
import time
from tqdm import tqdm

import sys
sys.path.append("./models")
sys.path.append("./data_utils")
sys.path.append("./losses")


from untied_model import deepfuse_triple_untied
from tied_model import deepfuse_triple_tied
from tied_model_double_encoder import deepfuse_triple_2encoder
from tied_model_triple_encoder import deepfuse_triple_3encoder
from ycbcr_ms_ssim import tf_ms_ssim
from termcolor import colored
import random
from transform_utils import *
from skimage.transform import resize
from prepare_batch import data_pipeline
import cv2
from vgg_utils import compute_percep_loss

begin_time = time.strftime("%m_%d-%H:%M:%S")
parser = argparse.ArgumentParser()
parser.add_argument('--train_patch_idx', default='./data_samples/fusion_train.txt', help='link to list of training patches')
parser.add_argument('--test_patch_idx', default='./data_samples/fusion_test.txt', help='link to list of test patches')
parser.add_argument('--fusion_model', default='triple_enc', help='tied|untied|double_enc|triple_enc')
parser.add_argument('--logdir', default='static_fusion_logs/', help='path to training logs')
parser.add_argument('--iters', default=200001, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--image_dim', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--restore', type=int, default=0, help='Train over checkpoint of train afresh')
parser.add_argument('--checkpoint_num', default=90001, help = 'which checkpoint to restore' )
parser.add_argument('--gpu', default=0, help = 'which gpu to use' )
parser.add_argument('--hdr', default=0, help = 'concatenate hdr along channels' )
parser.add_argument('--hdr_weight', default=1, help = 'weight for hdr loss')
parser.add_argument('--ssim_weight', default=5, help = 'weight for ssim loss')
parser.add_argument('--perceptual_weight', type=float, default=0.0001, help = 'weight for perceptual loss')



def restore_weights(sess, checkpoint_num):
    print('restoring weights from checkpoint num'+str(checkpoint_num))
    entire_saver = tf.train.Saver([v for v in tf.all_variables()])
    entire_saver.restore(sess, opts.logdir+'/checkpoints/model'+str(checkpoint_num)+'.ckpt')


def train(train_patch_list, val_patch_list, fusion_model, batch_size, image_dim, iters, checkpoint_dir, 
          train_logdir, val_logdir, restore, learning_rate, checkpoint_num, hdr, 
          hdr_weight, ssim_weight, perceptual_weight = 0):

    BATCH_SIZE = int(batch_size)
    IMAGE_DIM = int(image_dim)
    N_VAL_PATCHES = int(len(val_patch_list))

    if not (os.path.exists(train_logdir)):
        os.mkdir(train_logdir[:train_logdir.index('/')])
        os.mkdir(train_logdir)
    if not (os.path.exists(val_logdir)):
        os.mkdir(val_logdir)
    if not (os.path.exists(checkpoint_dir)):
        os.mkdir(checkpoint_dir)
    

    if hdr:
        im1 = tf.placeholder(tf.float32, shape=(None, None, None, 6))
        im2 = tf.placeholder(tf.float32, shape=(None, None, None, 6))
        im3 = tf.placeholder(tf.float32, shape=(None, None, None, 6))
    else:
        im1 = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        im2 = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        im3 = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    gt = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    train = tf.placeholder(tf.bool)

    if fusion_model == 'tied':
        output = deepfuse_triple_tied(im1, im2, im3)
    elif fusion_model == 'untied':
        output = deepfuse_triple_untied(im1, im2, im3)
    elif fusion_model == 'double_enc':
        output = deepfuse_triple_2encoder(im1, im2, im3)
    elif fusion_model == 'triple_enc':
        output = deepfuse_triple_3encoder(im1, im2, im3)
    else:
        print('Choose correct value for model type')
        return 
    compressed_gt = log_compressor(gt)
    compressed_output = log_compressor(output)

    y_ssim_loss = 1.0 - tf_ms_ssim(compressed_output, compressed_gt, batch_size=BATCH_SIZE, level=5)

    hdr_loss = tf.losses.absolute_difference(compressed_gt, compressed_output)

    pre_psnr = tf.image.psnr(gt, output, 1.0)
    post_psnr = tf.image.psnr(compressed_gt, compressed_output, 1.0)
    avg_pre_psnr = tf.reduce_mean(pre_psnr)
    avg_post_psnr = tf.reduce_mean(post_psnr)

    if perceptual_weight:
        perceptual_loss = tf.reduce_mean(compute_percep_loss(compressed_gt, compressed_output)*perceptual_weight)
    else:
        perceptual_loss = tf.constant(0.0)


    val_pre_psnr = tf.image.psnr(gt[:,12:-12, 18:-18,:], output[:,12:-12, 18:-18,:], 1.0)
    val_post_psnr = tf.image.psnr(compressed_gt[:,12:-12, 18:-18,:], compressed_output[:,12:-12, 18:-18,:], 1.0)
    val_avg_pre_psnr = tf.reduce_mean(val_pre_psnr)
    val_avg_post_psnr = tf.reduce_mean(val_post_psnr)

    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    overall_loss = (hdr_loss*hdr_weight)+(y_ssim_loss*ssim_weight)
    if perceptual_weight:
        overall_loss = overall_loss+perceptual_loss

    boundaries = [30000, 55000, 85000, 125000]
    values = [learning_rate, learning_rate/2.0, learning_rate/4.0, learning_rate/8.0, learning_rate/16.0]
    lr = tf.train.piecewise_constant(global_step, boundaries, values, 'lr_multisteps')
    print('total trainable vars :' + str(len(tf.trainable_variables())))

    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(lr)
        train_vars = [var for var in tf.trainable_variables() if 'DeepFuse' in var.name]
        train_step = optimizer.minimize(overall_loss, var_list=train_vars, global_step = global_step)

    y_loss_summary = tf.summary.scalar('y_ssim_tag', 1 - y_ssim_loss)

    pre_psnr_summary = tf.summary.scalar('pre_tonemapped_psnr_tag', avg_pre_psnr)
    post_psnr_summary = tf.summary.scalar('post_tonemapped_psnr_tag', avg_post_psnr)
    val_pre_psnr_summary = tf.summary.scalar('val_post_tonemapped_psnr_tag', val_avg_post_psnr)
    val_post_psnr_summary = tf.summary.scalar('val_pre_tonemapped_psnr_tag', val_avg_pre_psnr)
    perceptual_loss_summary = tf.summary.scalar('perceptual_loss', perceptual_loss)
    
    overall_summary = tf.summary.scalar('overall_loss_tag', overall_loss)

    inp1_im = tf.summary.image('training_image1', tf.cast(im1[:,:,:,:3]*255.0, tf.uint8))
    inp2_im = tf.summary.image('training_image2', tf.cast(im2[:,:,:,:3]*255.0, tf.uint8))
    inp3_im = tf.summary.image('training_image3', tf.cast(im3[:,:,:,:3]*255.0, tf.uint8))

    pred_im = tf.summary.image('train_pred', tf.cast(compressed_output*255.0, tf.uint8))
    gt_im = tf.summary.image('train_gt', tf.cast(compressed_gt*255.0, tf.uint8))

    loss_summary = tf.summary.merge(
        [pre_psnr_summary, post_psnr_summary, overall_summary, y_loss_summary, perceptual_loss_summary])
    image_summary = tf.summary.merge([inp1_im, inp2_im, inp3_im , pred_im, gt_im])

    val_inp1_im = tf.summary.image('validation_im1', tf.cast(im1[:,:,:,:3]*255.0, tf.uint8))
    val_inp2_im = tf.summary.image('validation_im2', tf.cast(im2[:,:,:,:3]*255.0, tf.uint8))
    val_inp3_im = tf.summary.image('validation_im3', tf.cast(im3[:,:,:,:3]*255.0, tf.uint8))

    val_pred_im = tf.summary.image('val_pred', tf.cast(compressed_output*255.0, tf.uint8))
    val_gt_im = tf.summary.image('val_gt', tf.cast(compressed_gt*255.0, tf.uint8))

    val_image_summary = tf.summary.merge([val_inp1_im, val_inp2_im, val_inp3_im, val_pred_im, val_gt_im])

    val_mse = tf.losses.mean_squared_error(output[:,12:-12,18:-18,:], gt[:,12:-12,18:-18,:])
    avg_val_mse = val_mse / 15.0
    val_summary = tf.summary.scalar('val_mse_tag', val_mse)
    val_loss_summary = tf.summary.merge([val_pre_psnr_summary, val_post_psnr_summary, val_summary])


    saver = tf.train.Saver(max_to_keep=200)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            restore_weights(sess, checkpoint_num)

        train_graph_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        val_graph_writer = tf.summary.FileWriter(val_logdir, sess.graph)

        train_dataset = data_pipeline(train_patch_list, BATCH_SIZE, IMAGE_DIM,hdr=hdr)
        train_iterator = train_dataset.make_initializable_iterator()
        train_batch = train_iterator.get_next()

        val_dataset = data_pipeline(val_patch_list, 1, IMAGE_DIM,val=True,hdr=hdr)
        val_iterator = val_dataset.make_initializable_iterator()
        val_batch = val_iterator.get_next()
        sess.run(val_iterator.initializer)
        sess.run(train_iterator.initializer)
        for i in range(iters):
            img1, img2, img3, ground_truth = sess.run(train_batch)
            gs, loss_write, image_write = sess.run([train_step, loss_summary, image_summary],
                                                       feed_dict={im1: img1, im2: img2,
                                                                  im3: img3,
                                                                  gt: ground_truth,
                                                                  train: True
                                                                  })
            train_graph_writer.add_summary(loss_write, global_step.eval())
            if (i % 30 == 0):
                train_graph_writer.add_summary(image_write, global_step.eval())
            if (i%10000 == 0):
                saver.save(sess, checkpoint_dir + 'model' + str(
                    global_step.eval())+'.ckpt')
            
                total_val_mse = 0
                total_pre_psnr = 0
                total_post_psnr = 0
                for val_batch_index in tqdm(range(0, N_VAL_PATCHES)):
                    img1, img2, img3, ground_truth = sess.run(val_batch)
                    mse_value, val_summ, val_image_write, pre_psnr_value, post_psnr_value = sess.run(
                                                                [val_mse, val_loss_summary, val_image_summary, val_avg_pre_psnr, val_avg_post_psnr],
                                                                feed_dict={im1: img1, im2: img2,
                                                                           im3: img3,
                                                                           gt: ground_truth,
                                                                           train: False})
                    total_val_mse += float(mse_value)
                    total_pre_psnr += float(pre_psnr_value)
                    total_post_psnr += float(post_psnr_value)

                    val_graph_writer.add_summary(val_summ, global_step.eval()+val_batch_index)
                    if (val_batch_index % 5 == 0):
                        val_graph_writer.add_summary(val_image_write, global_step.eval()+val_batch_index)
                total_val_mse = total_val_mse/N_VAL_PATCHES
                total_pre_psnr = total_pre_psnr/N_VAL_PATCHES
                total_post_psnr = total_post_psnr/N_VAL_PATCHES
                print(colored('mse: '+str(total_val_mse)+', pre-psnr: ' +str(total_pre_psnr)+', post-psnr: '
                          +str(total_post_psnr), 'green'))

        train_graph_writer.close()
        val_graph_writer.close()



if __name__ == '__main__':
    opts = parser.parse_args()
    train_idx_file = opts.train_patch_idx
    with open(train_idx_file, 'r') as f:
        train_patches = f.readlines()
        train_patches = [(x).rstrip() for x in train_patches]
    val_idx_file = opts.test_patch_idx
    with open(val_idx_file, 'r') as f:
        val_patches = f.readlines()
        val_patches = [(x).rstrip() for x in val_patches]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)

    train(train_patch_list=train_patches,
          val_patch_list=val_patches,
          fusion_model = opts.fusion_model,
          batch_size=opts.batch_size,
          restore=opts.restore,
          learning_rate=opts.lr,
          image_dim=opts.image_dim,
          checkpoint_dir=opts.logdir+'checkpoints/',
          iters=opts.iters,
          train_logdir=opts.logdir+'train/',
          val_logdir=opts.logdir+'val/',
          checkpoint_num = opts.checkpoint_num,
          hdr = opts.hdr,
          hdr_weight = int(opts.hdr_weight),
          ssim_weight = int(opts.ssim_weight),
          perceptual_weight =float( opts.perceptual_weight)
          )
