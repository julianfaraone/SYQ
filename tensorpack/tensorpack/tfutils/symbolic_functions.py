# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from ..utils import logger

def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
    """
    :param logits: NxC
    :param label: N
    :returns: a float32 vector of length N with 0/1 values. 1 means incorrect prediction
    """
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)),
            tf.float32, name=name)

def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.pack([tf.shape(x)[0], -1]))

def class_balanced_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    :param pred: size: b x ANYTHING. the predictions in [0,1].
    :param label: size: b x ANYTHING. the ground truth in {0,1}.
    :returns: class-balanced cross entropy loss
    """
    z = batch_flatten(pred)
    y = tf.cast(batch_flatten(label), tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    eps = 1e-12
    loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
    loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
    cost = tf.sub(loss_pos, loss_neg, name=name)
    return cost

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    This is more numerically stable than class_balanced_cross_entropy

    :param logits: size: the logits.
    :param label: size: the ground truth in {0,1}, of the same shape as logits.
    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)

    #logstable = tf.log(1 + tf.exp(-tf.abs(z)))
    #loss_pos = -beta * tf.reduce_mean(-y *
            #(logstable - tf.minimum(0.0, z)))
    #loss_neg = (1. - beta) * tf.reduce_mean((y - 1.) *
            #(logstable + tf.maximum(z, 0.0)))
    #cost = tf.sub(loss_pos, loss_neg, name=name)
    return cost

def print_stat(x, message=None):
    """ a simple print op.
        Use it like: x = print_stat(x)
    """
    if message is None:
        message = x.op.name
    return tf.Print(x, [tf.shape(x), tf.reduce_mean(x), x], summarize=20,
            message=message, name='print_' + x.op.name)

def rms(x, name=None):
    if name is None:
        name = x.op.name + '/rms'
        with tf.name_scope(None):   # name already contains the scope
            return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)

def huber_loss(x, delta=1, name='huber_loss'):
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.reduce_sum(
            tf.select(abscost < delta,
                sqrcost * 0.5,
                abscost * delta - 0.5 * delta ** 2),
            name=name)

def get_scalar_var(name, init_value, summary=False, trainable=False):
    """
    get a scalar variable with certain initial value
    :param summary: summary this variable
    """
    ret = tf.get_variable(name, shape=[],
            initializer=tf.constant_initializer(init_value),
            trainable=trainable)
    if summary:
        # this is recognized in callbacks.StatHolder
        tf.summary.scalar(name + '-summary', ret)
    return ret
