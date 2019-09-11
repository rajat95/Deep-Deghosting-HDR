import tensorflow as tf

import argparse
import os
import time
from tqdm import tqdm
import cv2
import sys

sys.path.append("./models")
sys.path.append("./data_utils")
sys.path.append("./losses")

from untied_model import deepfuse_triple_untied
from tied_model import deepfuse_triple_tied
from tied_model_double_encoder import deepfuse_triple_2encoder
from tied_model_triple_encoder import deepfuse_triple_3encoder
from refine_unet import refine_net
from PWCNet import pwc_net

from logCompressor import log_compressor
from transform_utils import *
from skimage.transform import resize
from tf_warp import backward_warp

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', default='./data_samples/test_data/', help='link to list of training patches')
parser.add_argument('--fusion_model', default='tied', help='tied|untied|double_encoder|triple_encoder')
parser.add_argument('--fusion_ckpt', default='./checkpoints/pretrained_tied_fusion/model150.ckpt', help='path to fusion ckpt')
parser.add_argument('--refine_ckpt', default='./checkpoints/pretrained_refine/model55.ckpt', help='path to refine ckpt')
parser.add_argument('--flow_ckpt', default='./checkpoints/pretrained_pwc/pwcnet.ckpt-595000', help='path to flow ckpt')
parser.add_argument('--ref_label', default=2, help = 'which image to use as reference')
parser.add_argument('--hdr_channels', default=1, help = 'weather to concatenate hdr along channels')
parser.add_argument('--gpu', default=0, help = 'which gpu to use')


def refine_dark(im1, im2, im12, im21, image_shape, batch_size=1, reuse=False, fl=None):
    """
    Called to generate darker version of reference image
    Args: 
        im1: Darker Under-exposed Image
        im2: Brighter reference image
        im21: im2 mapped to exposure of im1
        im12 : im1 mapped to exposure of im2
        fl: Optional flow if computed offline or using some other algorithm
    Returns: 
        output_im: corrected im21 
    """
    inp1 = tf.expand_dims(im2, 1)
    inp2 = tf.expand_dims(im12, 1)
    forward_vals = tf.concat([inp1, inp2], axis =1)
    backward_vals = tf.concat([inp2, inp1], axis = 1)
    if fl is not None:
        flow = fl
    else:
        pwc_inp = tf.concat([forward_vals, backward_vals], axis=0)
        flow, _ = pwc_net(pwc_inp)
    warped_im1 = backward_warp(im1, flow[:batch_size])
    warped_im1 = tf.reshape(warped_im1, tf.concat([image_shape,[3]], -1))
#    forward_occlusion, backward_occlusion = occlusion(flow[:batch_size], flow[batch_size:])
    refine_input = tf.concat([warped_im1, im21, flow[:batch_size]], axis=-1)
    output_map = refine_net(refine_input, name='dark_refine', reuse=reuse)
    output_im = output_map*im21+(1.-output_map)*warped_im1
    return output_im


def refine_bright(im1, im2, im21, image_shape, batch_size=1, reuse=False, fl=None):
    """
    Called to generate brighter version of reference image
    Args: 
        im1: Bright Over-exposed Image
        im2: Darker reference image
        im21: im2 mapped to exposure of im1
        fl: Optional flow if computed offline or using some other algorithm
    Returns: 
        output_im: corrected im21 
    """
    inp1 = tf.expand_dims(im21, 1)
    inp2 = tf.expand_dims(im1, 1)
    forward_vals = tf.concat([inp1, inp2], axis =1)
    backward_vals = tf.concat([inp2, inp1], axis = 1)
    if fl is not None:
        flow = fl
    else:
        pwc_inp = tf.concat([forward_vals, backward_vals], axis=0)
        flow, _ = pwc_net(pwc_inp)
    warped_im1 = backward_warp(im1, flow[:batch_size])
    warped_im1 = tf.reshape(warped_im1, tf.concat([image_shape,[3]], -1))
#    forward_occlusion, backward_occlusion = occlusion(flow[:batch_size], flow[batch_size:])
    refine_input = tf.concat([warped_im1, im21, flow[:batch_size]], axis=-1)
    output_map = refine_net(refine_input, name='dark_refine', reuse=reuse)
    output_im = output_map*im21+(1.-output_map)*warped_im1
    return output_im

def log_com(inp):
   return (tf.log(1+(5000.0*inp))/tf.log(1+5000.0))

def infer(source_dir, ref_label, fusion_model, fusion_ckpt, refine_ckpt, flow_ckpt):
    im1 = tf.placeholder(tf.float32,shape=[1,None,None,3])
    im2 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    im3 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    im23 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    im12 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    im21 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    im13 = tf.placeholder(tf.float32, shape=[1,None,None,3])
    image_dims = tf.placeholder(tf.int32, shape=[3])
    ev_bias = tf.placeholder(tf.float32, shape=[1])
   
    if ref_label==2:
        im1_ref = refine_dark(im1, im2, im12, im21, image_dims)
        im2_ref = im2
        im3_ref = refine_bright(im3, im2, im23, image_dims, reuse=True)
    if ref_label==1:
        im1_ref = im1
        im2_ref = refine_bright(im2, im1, im12, image_dims)
        im3_ref = refine_bright(im3, im1, im13, image_dims, reuse=True)
        
    if hdr_channels:
        im1_ref_hdr = tf_ldr_to_hdr(im1_ref, 2**(ev_bias*0))
        im2_ref_hdr = tf_ldr_to_hdr(im2_ref, 2**(ev_bias*1))
        im3_ref_hdr = tf_ldr_to_hdr(im3_ref, 2**(ev_bias*2))
        inp1 = tf.concat([im1_ref, im1_ref_hdr], axis=-1)
        inp2 = tf.concat([im2_ref, im2_ref_hdr], axis=-1)
        inp3 = tf.concat([im3_ref, im3_ref_hdr], axis=-1)
    else:
        inp1 = im1_ref
        inp2 = im2_ref
        inp3 = im3_ref
        
    if fusion_model == 'tied':
        final_hdr = deepfuse_triple_tied(inp1, inp2, inp3)
    elif fusion_model == 'untied':
        final_hdr = deepfuse_triple_untied(inp1, inp2, inp3)
    elif fusion_model == 'double_encoder':
        final_hdr = deepfuse_triple_2encoder(inp1, inp2, inp3)
    elif fusion_model == 'triple_encoder':
        final_hdr = deepfuse_triple_3encoder(inp1, inp2, inp3)
    else:
        print('Choose either of tied or untied model')
        return 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pwc_saver = tf.train.Saver([v for v in tf.all_variables() if "pwcnet" in v.name])
        pwc_saver.restore(sess, flow_ckpt)
        pwc_saver = tf.train.Saver([v for v in tf.all_variables() if "refine" in v.name])
        pwc_saver.restore(sess, refine_ckpt)
        saver = tf.train.Saver([v for v in tf.all_variables() if "DeepFuse"  in v.name])
        saver.restore(sess, fusion_ckpt)
        for path in os.listdir(source_dir):
            print('Generating HDR for ', path)
            img1 = np.expand_dims((cv2.imread(source_dir+path+'/1.tif', -1)[:,:,::-1].astype(np.float32))/65535.0, 0)
            img2 = np.expand_dims((cv2.imread(source_dir+path+'/2.tif', -1)[:,:,::-1].astype(np.float32))/65535.0, 0)
            img3= np.expand_dims((cv2.imread(source_dir+path+'/3.tif', -1)[:,:,::-1].astype(np.float32))/65535.0, 0)
            padding_x = 0
            padding_y = 0
            if (img1.shape[1])%64 != 0:
                padding_x = int(64*((img1.shape[1]//64)+1)-(img1.shape[1]))
            if (img1.shape[2])%64 != 0:
                padding_y = int(64*((img1.shape[2]//64)+1)-(img1.shape[2]))

            img1 = np.pad(img1, [[0,0],[padding_x//2, padding_x-(padding_x//2)], [padding_y//2, padding_y-(padding_y//2)], [0,0]], mode='reflect')
            img2 = np.pad(img2, [[0,0],[padding_x//2, padding_x-(padding_x//2)], [padding_y//2, padding_y-(padding_y//2)], [0,0]], mode='reflect')
            img3 = np.pad(img3, [[0,0],[padding_x//2, padding_x-(padding_x//2)], [padding_y//2, padding_y-(padding_y//2)], [0,0]], mode='reflect')

            with open(source_dir+path+'/exposure.txt') as f:
                lines = f.readlines()
                ev0 = float(lines[0][:-1])
                ev1 = float(lines[1][:-1])
                ev2 = float(lines[2][:-1])

            img12 = hdr_to_ldr(ldr_to_hdr(img1, 2**ev0), 2**ev1)
            img21 = hdr_to_ldr(ldr_to_hdr(img2, 2**ev1), 2**ev0)
            img23 = hdr_to_ldr(ldr_to_hdr(img2, 2**ev1), 2**ev2)
            img13 = hdr_to_ldr(ldr_to_hdr(img1, 2**ev0), 2**ev2)

            img1_ref, img2_ref, img3_ref, hdr_im = sess.run([im1_ref, im2_ref, im3_ref, final_hdr], feed_dict = {im1:img1, im2:img2,im3:img3, 
                                                                                                                 im12:img12, im21:img21, im23:img23, 
                                                                                                                 ev_bias: [ev1], image_dims:img1.shape[:-1]})
#            img1_ref = (img1_ref[0,padding_x//2:-1*(padding_x-(padding_x//2)),int(padding_y/2):-1*(padding_y-(padding_y//2)),:]*65535.0).astype(np.uint16)
#            img2_ref = (img2_ref[0,padding_x//2:-1*(padding_x-(padding_x//2)),int(padding_y/2):-1*(padding_y-(padding_y//2)),:]*65535.0).astype(np.uint16)
#            img3_ref = (img3_ref[0,padding_x//2:-1*(padding_x-(padding_x//2)),int(padding_y/2):-1*(padding_y-(padding_y//2)),:]*65535.0).astype(np.uint16)
            radiance_writer(source_dir+path+'/output.hdr', hdr_im[0,padding_x//2:-1*(padding_x-(padding_x//2)),int(padding_y/2):-1*(padding_y-(padding_y//2)),:])


if __name__ == '__main__':
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
    infer(source_dir = opts.source_dir, 
          ref_label = opts.ref_label,
          fusion_ckpt = opts.fusion_ckpt,
          fusion_model = opts.fusion_model,
          refine_ckpt = opts.refine_ckpt,
          flow_ckpt = opts.flow_ckpt)
   
