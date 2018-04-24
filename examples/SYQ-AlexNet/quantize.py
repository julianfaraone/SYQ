#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np

G = tf.get_default_graph()

def fine_grained_quant(x, eta, name, INITIAL, value, binary=True):
    
    shape = x.get_shape()

    eta_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * eta)

    list_of_masks = []

    if 'conv' in name:

        if INITIAL:
            w_s = tf.get_variable('Ws', [(shape[0].value*shape[1].value),1], collections=[tf.GraphKeys.VARIABLES, 'SCALING'], initializer=tf.constant_initializer(value[name + ':0']))
        else:
            w_s = tf.get_variable('Ws', [(shape[0].value*shape[1].value),1], collections=[tf.GraphKeys.VARIABLES, 'SCALING'], initializer=tf.constant_initializer(1.0))
        #scalar summary
        # for i in range(0,(shape[0].value*shape[1].value)):
        #     tf.scalar_summary(w_s.name + str(i) +str(0), w_s[i,0])
        
        #each pixel
        for i in range(shape[0].value):
            for j in range(shape[1].value):
                ws = w_s[(shape[1].value*i) + j, 0]
                mask = tf.ones(shape)
                mask_p = tf.select(x[i,j,:,:] > eta_x, mask[i,j,:,:] * ws, mask[i,j,:,:])
                mask_np = tf.select(x[i,j,:,:] < -eta_x, mask[i,j,:,:] * ws, mask_p)
                list_of_masks.append(mask_np)
                
        masker = tf.stack(list_of_masks)
        masker = tf.reshape(masker, [i.value for i in shape])

        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.select((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * masker

        tf.histogram_summary(w.name, w)
    else:

        if INITIAL:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.VARIABLES, 'scale_fc'], initializer=value[name + ':0'])
        else:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.VARIABLES, 'scale_fc'], initializer=1.0)

        #tf.scalar_summary(wp.name, wp)
        tf.scalar_summary(wn.name, wn)

        mask = tf.ones(shape)
        mask_p = tf.select(x > eta_x, tf.ones(shape) * wn, mask)
        mask_np = tf.select(x < -eta_x, tf.ones(shape) * wn, mask_p)
        
        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.select((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        tf.histogram_summary(w.name, w)


    return w

def rows_quant(x, eta, name, INITIAL, value, binary=True):

    shape = x.get_shape()

    eta_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * eta)

    list_of_masks = []

    if 'conv' in name:
        if INITIAL:
            w_s = tf.get_variable('Ws', [shape[0].value,1], collections=[tf.GraphKeys.VARIABLES, 'SCALING'], initializer=tf.constant_initializer(value[name + ':0']))
        else:
            w_s = tf.get_variable('Ws', [shape[0].value,1], collections=[tf.GraphKeys.VARIABLES, 'SCALING'], initializer=tf.constant_initializer(1.0))

        for i in range(0,(shape[0].value*shape[1].value)):
            tf.scalar_summary(w_s.name + str(i) +str(0), w_s[i,0])
            tf.scalar_summary(w_s.name + str(i) + str(1), w_s[i, 1])
        
        #each row
        for j in range(shape[0].value):
            ws = w_s[j , 0]
            mask = tf.ones(shape)
            mask_p = tf.select(x[i,:,:,:] > eta_x, mask[i,:,:,:] * ws, mask[i,:,:,:])
            mask_np = tf.select(x[i,:,:,:] < -eta_x, mask[i,:,:,:] * ws, mask_p)
            list_of_masks.append(mask_np)

        masker = tf.stack(list_of_masks)
        masker = tf.reshape(masker, [i.value for i in shape])

        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.select((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * masker

        tf.histogram_summary(w.name, w)

    else:
        if INITIAL:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.VARIABLES, 'scale_fc'], initializer=value[name + ':0'])
        else:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.VARIABLES, 'scale_fc'], initializer=1.0)

        tf.scalar_summary(wn.name, wn)

        mask = tf.ones(shape)
        mask_p = tf.select(x > eta_x, tf.ones(shape) * wn, mask)
        mask_np = tf.select(x < -eta_x, tf.ones(shape) * wn, mask_p)
        
        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.select((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        tf.histogram_summary(w.name, w)

    return w

def quantize(x, k, fraclength=None, signed=True):
    if fraclength != None:
        f = fraclength
        n = float(2.**f)
        mn = - 2.**(k - f - 1)
        mx = -mn - 2.**-f
        if not signed:
            mx -= mn
            mn = 0
        x = tf.clip_by_value(x, mn, mx)
    else:
        n = float(2**k-1)
    with G.gradient_override_map({"Floor": "Identity"}):
        return tf.floor(x * n + 0.5) / n
