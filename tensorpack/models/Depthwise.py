#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: Depthwis2D.py
# Author: Julian Faraone <julian.faraone@sydney.edu.au>

import numpy as np
import tensorflow as tf
import math
from ._common import layer_register, shape2d, shape4d
from ..utils import logger
from ..utils.argtools import shape2d

#when this module is imported, the depthwise_separable_conv symbol(function) will be exported
__all__ = ['Depthwise']

@layer_register()
def Depthwise(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.identity, channel_multiplier=1, use_bias=True,
           data_format='NHWC'):
    """ Function to build the depth-wise convolution layer."""
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Depthwise] Input cannot have unknown channel!"

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel, channel_multiplier]
    stride = [1, stride, stride, 1]

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    depth = tf.nn.depthwise_conv2d(x, W, stride, padding, name='depthwise_weights')

    return depth

