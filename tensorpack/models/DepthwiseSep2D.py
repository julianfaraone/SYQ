
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: DepthwiseSep2D.py
# Author: Julian Faraone <julian.faraone@sydney.edu.au>

import numpy as np
import tensorflow as tf
import math
from ._common import layer_register
from ..utils import logger

#slim allows us to combine convolutions, batch norm and relu into one function
slim = tf.contrib.slim

#when this module is imported, the depthwise_separable_conv symbol(function) will be exported
__all__ = ['depthwise_separable_conv']

@layer_register()
def depthwise_separable_conv(x, num_pwc_filters,
                                kernel_size,
                                stride,
                                depth_multiplier=1,
                                padding="SAME",
                                rate=1,
                                scope=None):
    """ Function to build the depth-wise separable convolution layer.
    """
    #num_pwc_filters stands for number of pointwise convolutional filters.
    num_pwc_filters = round(num_pwc_filters * depth_multiplier)

    #depth_multiplier is equivalent to width_multiplier in paper
    # skip pointwise by setting num_outputs=None
    batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': 0.9997,
      'epsilon': 0.001,
  }
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):

      depthwise_conv = slim.separable_convolution2d(x,
                                                    num_outputs=None,
                                                    stride=stride,
                                                    depth_multiplier=1,
                                                    padding="SAME",
                                                    normalizer_fn=slim.batch_norm,
                                                    activation_fn=tf.nn.relu,
                                                    kernel_size=kernel_size,
                                                    scope='dw')

      pointwise_conv = slim.convolution2d(depthwise_conv,
                                          num_pwc_filters,
                                          kernel_size=[1, 1],
                                          padding="SAME",
                                          stride=1,
                                          normalizer_fn=slim.batch_norm,
                                          activation_fn=tf.nn.relu,
                                          scope='sep')

    return pointwise_conv
