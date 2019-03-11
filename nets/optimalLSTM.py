from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops


from pprint import pprint


def _conv(args, filter_size, num_features, bias, bias_start=0.0):
  """Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length * [1]

  # Now the computation.

  optimalW = np.zeros(filter_size + [total_arg_size_depth, num_features])
  optimalB = np.zeros([num_features])
  #in gate
  optimalW[1, 1, 0, 0] = -1e4
  optimalW[1, 1, 0, 1] = -1e4
  optimalB[0] = 5e3
  optimalB[1] = 5e3

  #C cell gate:
  optimalB[2] = -1e4
  optimalB[3] = 1e4

  #forget gate
  optimalB[4] = -1e5
  optimalB[5] = -1e5

  #output gate:

  optimalW[0, :, 2, 6] = 1e5
  optimalW[1, 0, 2, 6] = 1e5
  optimalW[1, 2, 2, 6] = 1e5
  optimalW[2, :, 2, 6] = 1e5
  optimalW[:, :, :, 7] = optimalW[:, :, :, 6]

  optimalB[6] = -5e4
  optimalB[7] = -5e4


  kernel = tf.get_variable(
      "kernel",
      initializer=tf.constant(optimalW, dtype=tf.float32))

  input = array_ops.concat(axis=shape_length - 1, values=args)
  padded_input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT", constant_values=1.0)

  res = tf.nn.conv2d(input=padded_input,
                     filter=kernel, strides=[1, 1, 1, 1], padding="VALID")

  bias_term = vs.get_variable(
      "biases",
      initializer=tf.constant(optimalB, dtype=tf.float32))

  return res + bias_term


class ConvLSTMCell_tunned(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               input_shape,
               output_channels,
               kernel_shape,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTMCell_tunned, self).__init__(name=name)


    self._conv_ndims = 2
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    cell, hidden = state
    new_hidden = _conv([inputs, hidden], self._kernel_shape,
                       4 * self._output_channels, self._use_bias)
    gates = array_ops.split(
        value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)

    input_gate, new_input, forget_gate, output_gate = gates
    new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
    new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
    output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = array_ops.concat([output, inputs], axis=-1)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state


def optimalLSTM(data, opt, dropout_rate, labels_id):
    """ Run the coloring network on data, with hyperparameters

    :param data: in the shape batch_size, image_height, image_width, 2
        where the last two channels are the inside and outside contour
    :param opt:
        opt.hyper parameters are
        n_t number of timepoints for the network to run
        n_hidden number of hidden layers
    :param dropout_rate:
    :param labels_id:
    :return:
    """

    n_t = getattr(opt.dnn, "n_t", 10)

    data = tf.reshape(data,
                      [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    cell1 = ConvLSTMCell_tunned(input_shape=data.shape.dims[-3:],
                          kernel_shape=[3, 3],
                          output_channels=2, name='a')

    istate1_zero = tf.constant(np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 2]),
                               dtype=np.float32)
    istate1 = np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 2])
    istate1[:, 0, :, :] = 1
    istate1[:, data.shape[1] - 1, :, :] = 1
    istate1[:, :, 0, :] = 1
    istate1[:, :, data.shape[2] - 1, :] = 1
    iistate1 = tf.constant(istate1, dtype=np.float32)

    state1 = tf.nn.rnn_cell.LSTMStateTuple(istate1_zero, iistate1)

    out = []
    act_state1 = []
    with tf.variable_scope("scp") as scope:
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()

            t_output, state1 = cell1(data, state1)

            out.append([t_output[:, :, :, :2]])
            act_state1.append(state1)


    if opt.dnn.train_per_step:
        return out, [cell1.weights], [act_state1]
    else:
        return out[-1], [], [act_state1]
