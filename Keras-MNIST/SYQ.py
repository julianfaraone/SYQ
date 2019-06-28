from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

import tensorflow as tf
import tensorflow.keras as keras

def quantize(x, bit_width):
        if bit_width == 1:
                return x + tf.stop_gradient(tf.sign(x) - x)
        else:
                return x

class SYQ(Conv2D):

  def __init__(self, bit_width, *args, **kwargs):
        self.bit_width = bit_width
        super(SYQ, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)
 
    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    #add learnable scaing factor
    self.scale = self.add_weight('scale' ,shape=kernel_shape, initializer=keras.initializers.Ones(), dtype=self.dtype, trainable=True)

    #quantize weights
    self.kernel = quantize(self.kernel, self.bit_width) * self.scale

    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=op_padding,
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2))
    self.built = True

class SYQ_Dense(Dense):

  def __init__(self, bit_width, *args, **kwargs):
        self.bit_width = bit_width
        super(SYQ_Dense, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    #add learnable scaing factor
    self.scale = self.add_weight('scale' ,shape=[1], initializer=keras.initializers.Ones(), dtype=self.dtype, trainable=True)

    #quantize weights
    self.kernel = quantize(self.kernel, self.bit_width) * self.scale

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

