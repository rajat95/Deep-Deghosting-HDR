from scipy.ndimage.interpolation import affine_transform as afft
import numpy as np
import random
import tensorflow as tf

def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def center_crop(ccrop_img, crop_size):
    height = ccrop_img.shape[0]
    width = ccrop_img.shape[1]
    startx = width // 2 - (crop_size // 2)
    starty = height // 2 - (crop_size // 2)
    return ccrop_img[starty: starty + crop_size, startx: startx + crop_size]


def random_crop(img, width, height, x_limit = 0, y_limit = 0):
    # CAUTION: TAKES IN A BUNCH OF IMAGES AS A 4D ARRAY
    assert len(img.shape) == 4
    assert img.shape[1] >= height
    assert img.shape[2] >= width

    x = random.randint(x_limit, img.shape[2] - width-x_limit)
    y = random.randint(y_limit, img.shape[1] - height-y_limit)

    img = img[:, y:y+height, x:x+width]
    return img


def affine_transform(img, affine_mat):
    r_transformed = afft(img[:, :, 0], affine_mat)
    g_transformed = afft(img[:, :, 1], affine_mat)
    b_transformed = afft(img[:, :, 2], affine_mat)
    return np.stack([r_transformed, g_transformed, b_transformed], axis = -1)

def ldr_to_hdr(im, t, gamma = 2.2):
    im_out = im**gamma
    im_out = im_out/t
    return im_out

def hdr_to_ldr(im, t, gamma = 2.2):
    im_out = im*t
    im_out = np.clip(im_out**(1.0/gamma),0,1)
    return im_out

def tf_ldr_to_hdr(im, t, batch_size = 1):
    t = tf.reshape(t, [batch_size, 1, 1, 1])
    out = tf.pow(im, 2.2)
    out = out/t
    return out

def log_compressor(im, MU = 5000.):
    return tf.log(1+MU*im)/tf.log(1+MU)

def log_compressor_np(im, MU = 5000.):
    return np.log(1+MU*im)/np.log(1+MU)
