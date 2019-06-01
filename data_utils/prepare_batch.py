import tensorflow as tf
import numpy as np
import random
import cv2
from skimage.transform import resize
from affine import Affine
from transform_utils import *


def load_train_patch(idx, index_list,  BATCH_SIZE, IMAGE_DIM, hdr):
    im1 = cv2.imread(index_list[idx].decode('utf8')+'/refined_result/2_1.tif', -1).astype(np.float32)
    im1 = (im1[:, :, ::-1]/65535.0).astype(np.float32)
    im2 = cv2.imread(index_list[idx].decode('utf8')+'/2.tif', -1).astype(np.float32)
    im2 = (im2[:, :, ::-1]/65535.0).astype(np.float32)
    im3 = cv2.imread(index_list[idx].decode('utf8')+'/refined_result/2_3.tif', -1).astype(np.float32)
    im3 = (im3[:, :, ::-1]/65535.0).astype(np.float32)
   
#    print(gt_path)
    gt = cv2.imread(index_list[idx].decode('utf8')+'/HDRImg.hdr', -1).astype(np.float32)
    gt = gt[:,:,::-1]
    #gt = (gt[:, :, ::-1]/255.0).astype(np.float32)
    

    crop_dim = np.random.choice([1,2,3]) #take 256*3 sized crops
    rotation_tup = (-3, 3)
    translation_tup = (-0.05, 0.05)
   
    output_arr = random_crop(np.asarray([im1, im2, im3, gt]), int(IMAGE_DIM*crop_dim), int(IMAGE_DIM*crop_dim), 100, 100)
    output_arr = resize(output_arr, [4, IMAGE_DIM, IMAGE_DIM , 3], anti_aliasing=False)
    rand = random.random()
    if rand>0.5:
        output_arr = output_arr[:, ::-1, :, :]
    rand = random.random()
    if rand>0.5:
        output_arr = output_arr[:, :, ::-1, :]

    rot_count = np.random.choice([0,1, 2, 3])
    output_arr = np.rot90(output_arr, k=rot_count, axes = (1, 2))
    random_order = np.random.permutation(3)
    output_arr = output_arr[:,:,:,random_order]

    if hdr:
        with open(index_list[idx].decode('utf8')+'/exposure.txt') as f:
            exp = float(f.readlines()[1][:-1])
        e_t0 = 1.0
        e_t1 = 2.0**exp
        e_t2 = 2.0**(2*exp)
    
        hdr1 = ldr_to_hdr(output_arr[0],e_t0)
        hdr2 = ldr_to_hdr(output_arr[1], e_t1)
        hdr3 = ldr_to_hdr(output_arr[2], e_t2)
        return np.concatenate([output_arr[0], hdr1], axis=-1), np.concatenate([output_arr[1], hdr2], axis=-1), np.concatenate([output_arr[2], hdr3], axis=-1), output_arr[3]
    return output_arr[0], output_arr[1], output_arr[2], output_arr[3]


def load_val_patch(idx, index_list,  BATCH_SIZE, IMAGE_DIM, hdr):
    im1 = cv2.imread(index_list[idx].decode('utf8')+'/refined_result/2_1.tif', -1).astype(np.float32)
    im1 = (im1[:, :, ::-1]/65535.0).astype(np.float32)
    im2 = cv2.imread(index_list[idx].decode('utf8')+'/2.tif', -1).astype(np.float32)
    im2 = (im2[:, :, ::-1]/65535.0).astype(np.float32)
    im3 = cv2.imread(index_list[idx].decode('utf8')+'/refined_result/2_3.tif', -1).astype(np.float32)
    im3 = (im3[:, :, ::-1]/65535.0).astype(np.float32)
#    print(gt_path)
    im1 = np.pad(im1, [(12,12),(18,18),(0,0)], mode='reflect') 
    im2 = np.pad(im2, [(12,12),(18,18),(0,0)], mode='reflect')
    im3 = np.pad(im3, [(12,12),(18,18),(0,0)], mode='reflect')

    gt = cv2.imread(index_list[idx].decode('utf8')+'/HDRImg.hdr', -1).astype(np.float32)
    gt = gt[:,:,::-1]
    gt = np.pad(gt, [(12,12),(18,18),(0,0)], mode='reflect')
    #gt = (gt[:, :, ::-1]/255.0).astype(np.float32)
    if hdr:
        with open(index_list[idx].decode('utf8')+'/exposure.txt') as f:
            exp = float(f.readlines()[1][:-1])
        e_t0 = 1.0
        e_t1 = 2.0**exp
        e_t2 = 2.0**(2*exp)
    
        hdr1 = ldr_to_hdr(im1,e_t0)
        hdr2 = ldr_to_hdr(im2, e_t1)
        hdr3 = ldr_to_hdr(im3, e_t2)
        return np.concatenate([im1, hdr1], axis=-1), np.concatenate([im2, hdr2], axis=-1), np.concatenate([im3, hdr3], axis=-1), gt

    return im1, im2, im3, gt


def data_pipeline(idx_list, batch_size, input_dim,val=False, hdr=True):
    index = np.arange(len(idx_list))
    dataset = tf.data.Dataset.from_tensor_slices(index)
    dataset = dataset.shuffle(len(idx_list)).repeat(-1)
    if val:
        dataset = dataset.map(lambda idx: tf.py_func(load_val_patch, [idx, idx_list, batch_size, input_dim,hdr], [tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=16)
    else:
        dataset = dataset.map(lambda idx: tf.py_func(load_train_patch, [idx, idx_list, batch_size, input_dim,hdr], [tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def get_refine_batch(idx, index_list, BATCH_SIZE, IMAGE_DIM):
    im1 = cv2.imread(index_list[idx].decode('utf8')+'img1.tif', -1).astype(np.float32)
    im1 = (im1[:, :, ::-1]/65535.0).astype(np.float32)
    im2 = cv2.imread(index_list[idx].decode('utf8')+'img2.tif', -1).astype(np.float32)
    im2 = (im2[:, :, ::-1]/65535.0).astype(np.float32)
    im3 = cv2.imread(index_list[idx].decode('utf8')+'img3.tif', -1).astype(np.float32)
    im3 = (im3[:, :, ::-1]/65535.0).astype(np.float32)
    gt_path = index_list[idx].decode('utf8').replace('ghosted', 'reference')

    gt2 = cv2.imread(gt_path+'img1.tif', -1).astype(np.float32)
    gt2 = (gt2[:, :, ::-1] / 65535.0).astype(np.float32)

    gt3 = cv2.imread(gt_path+'img3.tif', -1).astype(np.float32)
    gt3 = (gt3[:, :, ::-1] / 65535.0).astype(np.float32)

    gt_path = index_list[idx].decode('utf8').replace('input', 'fused_TM').replace('ghosted', '')

    gt = cv2.imread(gt_path+'fused.png', -1).astype(np.float32)
#    gt = (gt[:, :, ::-1]/255.0).astype(np.float32)
    im12 = cv2.imread(gt_path+'img12.tif', -1).astype(np.float32)
    im12 = (im12[:, :, ::-1]/65535.0).astype(np.float32)
    im23 = cv2.imread(gt_path+'img23.tif', -1).astype(np.float32)
    im23 = (im23[:, :, ::-1]/65535.0).astype(np.float32)
    im21 = cv2.imread(gt_path+'img21.tif', -1).astype(np.float32)
    im21 = (im21[:, :, ::-1]/65535.0).astype(np.float32)
    
    fl_path = gt_path.replace('fused_TM', 'flows')
    fl1 = np.load(fl_path+'flow_21.npy')[:,:,:2]
    fl2 = np.load(fl_path+'flow_23.npy')[:,:,:2]
    crop_dim = 1 
    rotation_tup = (-3, 3)
    translation_tup = (-0.05, 0.05)

    height, width,_ = im2.shape

    output_arr = np.asarray([im1, im2, im3, im12, im21, im23, gt2, gt3])
    output_fl = np.asarray([fl1, fl2])

    x_limit = 100
    y_limit = 100
    width = IMAGE_DIM*crop_dim
    height = IMAGE_DIM*crop_dim
    x = random.randint(x_limit, output_arr.shape[2] - width-x_limit)
    y = random.randint(y_limit, output_arr.shape[1] - height-y_limit)

    output_arr = output_arr[:, y:y+height, x:x+width]
    output_fl = output_fl[:, y:y+height, x:x+width]


    rand = random.random()
    if rand>0.5:
        output_arr = output_arr[:, ::-1, :, :]
        output_fl = output_fl[:,::-1,:,:]
        output_fl[:,:,:,0] = -1*output_fl[:,:,:,0]
    rand = random.random()
    if rand>0.5:
        output_arr = output_arr[:, :, ::-1, :]
        output_fl = output_fl[:,:,::-1,:]
        output_fl[:,:,:,1] = -1*output_fl[:,:,:,1]
    rand = random.random()
    if rand>0.5:
        rot_count = np.random.choice([1, 2, 3])
        output_arr = np.rot90(output_arr, k=rot_count, axes = (1, 2))
        output_fl = np.rot90(output_fl, k=rot_count, axes = (1, 2))
        output_fl_ = output_fl
        k = rot_count
        output_fl[:,:,:,0] = np.sin(k*np.pi/2.0)*output_fl_[:,:,:,1]+np.cos(k*np.pi/2.0)*output_fl_[:,:,:,0]
        output_fl[:,:,:,1] = -np.sin(k*np.pi/2.0)*output_fl_[:,:,:,0]+np.cos(k*np.pi/2.0)*output_fl_[:,:,:,1]
    random_order = np.random.permutation(3)
    output_arr = output_arr[:,:,:,random_order]
    return output_arr[0], output_arr[1], output_arr[2], output_arr[3], output_arr[4], output_arr[5], output_arr[6], output_arr[7], output_fl[0], output_fl[1]
    

def data_pipeline_refine(idx_list, batch_size, input_dim, patch_type = 'im'):
    index = np.arange(len(idx_list))
    dataset = tf.data.Dataset.from_tensor_slices(index)
    dataset = dataset.shuffle(len(idx_list)).repeat(-1)
    dataset = dataset.map(lambda idx: tf.py_func(get_refine_batch, [idx, idx_list, batch_size, input_dim], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

