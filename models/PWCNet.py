#PWC Net implementation borrowed from repo here https://github.com/philferriere/tfoptflow 
#Written by Phil Ferriere

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

PWC_NET_VARS = ['featpyrs', 'warp', 'corr', 'predict_flow', 'up_flow', 'ctxt', 'pwcnet']


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = array_ops.unstack(array_ops.shape(grid))
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received: '
            raise ValueError(msg + str(shape))

        batch_size, height, width, channels = shape
        query_type = query_points.dtype
        query_shape = array_ops.unstack(array_ops.shape(query_points))
        grid_type = grid.dtype

        if len(query_shape) != 3:
            msg = ('Query points must be 3 dimensional. Received: ')
            raise ValueError(msg + str(query_shape))

        _, num_queries, _ = query_shape

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        flattened_grid = array_ops.reshape(grid,
                                           [batch_size * height * width, channels])
        batch_offsets = array_ops.reshape(
            math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


def dense_image_warp(image, flow, name='dense_image_warp'):
    with ops.name_scope(name):
        batch_size, height, width, channels = array_ops.unstack(array_ops.shape(image))
        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = array_ops.meshgrid(
            math_ops.range(width), math_ops.range(height))
        stacked_grid = math_ops.cast(
            array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = array_ops.reshape(query_points_on_grid,
                                                   [batch_size, height * width, 2])
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = _interpolate_bilinear(image, query_points_flattened)
        interpolated = array_ops.reshape(interpolated,
                                         [batch_size, height, width, channels])
        return interpolated


def extract_features(x_tnsr,reuse_val = False, name='featpyr'):
    num_channels = [None, 16, 32, 64, 96, 128, 196]
    c1, c2 = [None], [None]
    if reuse_val:
        reuse_arr = [True, True]
    else:
        reuse_arr = [None, True]
    init = tf.contrib.keras.initializers.he_normal()
    with tf.variable_scope(name, reuse_val):
        for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
            for lvl in range(1, 6 + 1):
                # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                # reuse is set to True because we want to learn a single set of weights for the pyramid
                # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                f = num_channels[lvl]
                x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name='conv{}a'.format(lvl), reuse=reuse)
                x = tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv{}aa'.format(lvl), reuse=reuse)
                x =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)  # , name=f'relu{lvl+1}aa')
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv{}b'.format(lvl), reuse=reuse)
                x =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
                pyr.append(x)
    return c1, c2


    ###
    # PWC-Net warping helpers
    ###
def warp(c2, sc_up_flow, lvl='final', name='warp'):
    op_name = '{}{}'.format(name, lvl)
    with tf.name_scope(name):
        return dense_image_warp(c2, sc_up_flow, name=op_name)

def deconv(x, lvl, name='up_flow'):
        op_name = '{}{}'.format(name, lvl)
        with tf.variable_scope('upsample'):
            # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
            return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)


def cost_volume(c1, warp, search_range, name):

    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keep_dims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(cost_vol)

    return cost_vol

def corr(c1, warp, lvl, name='corr'):
    op_name = 'corr{}'.format(lvl)
    with tf.name_scope(name):
        return cost_volume(c1, warp, 4, op_name)


def predict_flow( corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
    op_name = 'flow{}'.format(lvl)
    init = tf.contrib.keras.initializers.he_normal()
    with tf.variable_scope(name):
        if c1 is None and up_flow is None and up_feat is None:
            x = corr
        else:
            x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv{}_0'.format(lvl))
        act =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(conv)  # default alpha is 0.2 for TF
        x = tf.concat([act, x], axis=3)

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv{}_1'.format(lvl))
        act =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(conv)
        x = tf.concat([act, x], axis=3)

        conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name='conv{}_2'.format(lvl))
        act =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(conv)
        x = tf.concat([act, x], axis=3)

        conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name='conv{}_3'.format(lvl))
        act =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(conv)
        x = tf.concat([act, x], axis=3)

        conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name='conv{}_4'.format(lvl))
        act =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(conv)  # will also be used as an input by the context network
        upfeat = tf.concat([act, x], axis=3, name='upfeat{}'.format(lvl))

        flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

        return upfeat, flow

def refine_flow(feat, flow, lvl, name='ctxt'):
    op_name = 'refined_flow{}'.format(lvl)
    init = tf.contrib.keras.initializers.he_normal()
    with tf.variable_scope(name):
        x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}1'.format(lvl))
        x =  tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)  # default alpha is 0.2 for TF
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name='dc_conv{}2'.format(lvl))
        x = tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name='dc_conv{}3'.format(lvl))
        x =tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name='dc_conv{}4'.format(lvl))
        x = tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name='dc_conv{}5'.format(lvl))
        x =tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}6'.format(lvl))
        x = tf.contrib.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}7'.format(lvl))

        return tf.add(flow, x, name=op_name)


def pwc_net(x_tnsr, name='pwcnet', pyr_lvls = 6, flow_pred_lvl = 2, reuse = None):


            # Extract pyramids of CNN features from both input images (1-based lists))
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) :

            # Extract pyramids of CNN features from both input images (1-based lists))
        c1, c2 = extract_features(x_tnsr, reuse_val = reuse)

        flow_pyr = []

        for lvl in range(pyr_lvls, flow_pred_lvl - 1, -1):
            if lvl == pyr_lvls:
                    # Compute the cost volume
                corr_feats = corr(c1[lvl], c2[lvl], lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr_feats, None, None, None, lvl)
            else:
                # Warp level of Image1's using the upsampled flow
                scaler = 20. / 2**lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                warp_feats = warp(c2[lvl], up_flow * scaler, lvl)

                # Compute the cost volume
                corr_feats = corr(c1[lvl], warp_feats, lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr_feats, c1[lvl], up_flow, up_feat, lvl)

            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

            if lvl != flow_pred_lvl:

                flow = refine_flow(upfeat, flow, lvl)

                # Upsample predicted flow and the features used to compute predicted flow
                flow_pyr.append(flow)

                up_flow = deconv(flow, lvl, 'up_flow')
                up_feat =deconv(upfeat, lvl, 'up_feat')
            else:
                # Refine the final predicted flow
                flow = refine_flow(upfeat, flow, lvl)
                flow_pyr.append(flow)

                # Upsample the predicted flow (final output) to match the size of the images
                scaler = 2**flow_pred_lvl

                size = (lvl_height * scaler, lvl_width * scaler)
                flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scaler
                break

    return flow_pred, flow_pyr
