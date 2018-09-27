from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops

from pprint import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class FillingCell(RNNCell):
    """ A tensorflow recurrent neural network cell which
    implements flood-filling

    """
    def __init__(self, input_shape, optimal=True, n_hidden=2, weight_std=1):
        super().__init__()
        self._input_shape = input_shape
        self._state_size = tensor_shape.TensorShape(
            list(self._input_shape[1:-1]) + [2])
        self._output_size = self.state_size[1:]
        self._optimal = optimal
        self._weight_std = weight_std
        self._n_hidden = n_hidden

        if self._optimal:
            full_kernel = np.zeros((3, 3, 1, 1))
            a_kernel = np.array([[0, -1,  0],
                                 [-1, 0, -1],
                                 [0, -1,  0]])
            full_kernel[:, :, 0, 0] = a_kernel
            self._kernel = tf.constant(full_kernel, dtype=tf.float32)
            self._bias_i = tf.constant(1., dtype=tf.float32)
            self._w1 = tf.constant(-1., dtype=tf.float32)
            self._w2 = tf.constant(-1., dtype=tf.float32)
            self._bias_s = tf.constant(1., dtype=tf.float32)
        else:
            self._kernel = tf.Variable(tf.truncated_normal((3, 3, 1, 1),
                                               dtype=tf.float32,
                                               stddev=self._weight_std,
                                               name="W"
                                               ))
            self._bias_i = tf.Variable(0.1, name="bias_a", dtype=tf.float32)
            self._w1 = tf.Variable(0.1, name="w1", dtype=tf.float32)
            self._w2 = tf.Variable(0.1, name="w2", dtype=tf.float32)
            self._bias_s = tf.Variable(0.1, name="bias_s", dtype=tf.float32)


    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size


    def get_params(self):
        if self._optimal:
            return []
        return [self._kernel, self._bias_i, self._w1, self._w2, self._bias_s]

    def call(self, border, state):
        intermediate = tf.nn.relu(
            nn_ops.conv2d(state, self._kernel, [1, 1, 1, 1], padding="SAME")
            +
            self._bias_i
        )

        state = tf.nn.relu(
            self._bias_s + self._w1 * border + self._w2 * intermediate)
        return 1-state, state



def Coloring(data, opt, dropout_rate, labels_id):
    """ Run the coloring network on data, with hyperparameters

    :param data: in the shape batch_size, image_height, image_width, 2
        where the last two channels are the inside and outside contour
    :param opt:
        opt.optimal if true, run the designed model, which will not be trained
        opt.hyper parameters are
        n_t number of timepoints for the network to run
        n_hidden number of hidden layers
        weight_std for initialization the weight standard deviation
    :param dropout_rate:
    :param labels_id:
    :return:
    """

    optimal = getattr(opt, "skip_train", True)
    fc = FillingCell(input_shape=data.shape,
                     optimal=optimal,
                     n_hidden=getattr(opt.dnn, "layers", 2),
                     weight_std=getattr(opt.dnn, "neuron_multiplier", 2),
                     )
    if not optimal:
        parameters = fc.get_params()
    else:
        parameters = []

    n_t = opt.dnn.n_t

    data = tf.reshape(data, [-1, data.shape[1], data.shape[2], 1])

    activations = []

    initial_state = np.zeros(data.shape[1:3])
    initial_state[0, :] = 1
    initial_state[data.shape[1] - 1, :] = 1
    initial_state[:, 0] = 1
    initial_state[:, data.shape[2] - 1] = 1
    state = tf.constant(initial_state[None, :, :, None], dtype=np.float32)

    with tf.variable_scope("FilledCell") as scope:
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()
            t_output, state = fc(data, state)
            activations.append(state)

    return tf.concat([t_output, state], 3), parameters, activations
