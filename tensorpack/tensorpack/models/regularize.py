# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

from ..utils import logger
from ..utils.argtools import memoized
from ..tfutils.tower import get_current_tower_context
from ._common import layer_register

__all__ = ['regularize_cost', 'l2_regularizer', 'l1_regularizer', 'Dropout']

@memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))

l2_regularizer = tf.contrib.layers.l2_regularizer
l1_regularizer = tf.contrib.layers.l1_regularizer

def regularize_cost(regex, func, name=None):
    """
    Apply a regularizer on every trainable variable matching the regex.

    :param func: a function that takes a tensor and return a scalar.
    """
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    costs = []
    for p in params:
        para_name = p.name
        if re.search(regex, para_name):
            costs.append(func(p))
            _log_regularizer(para_name)
    if not costs:
        return 0
    return tf.add_n(costs, name=name)


@layer_register(log_shape=False, use_scope=False)
def Dropout(x, keep_prob=0.5, is_training=None):
    """
    :param is_training: if None, will use the current context by default.
    """
    if is_training is None:
        is_training = get_current_tower_context().is_training
    keep_prob = tf.constant(keep_prob if is_training else 1.0)
    return tf.nn.dropout(x, keep_prob)

