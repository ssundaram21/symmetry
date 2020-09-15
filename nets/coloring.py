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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class FillingCell(RNNCell):
    """ A tensorflow recurrent neural network cell which
    implements flood-filling

    """

    def __init__(self, input_shape, optimal=True, n_hidden=2, weight_std=1.0, range_coloring = 3):
        super().__init__()
        self._input_shape = input_shape
        self._state_size = tensor_shape.TensorShape(
            list(self._input_shape[1:-1]) + [2])
        self._output_size = self._state_size[1:]
        self._optimal = optimal
        self._weight_std = weight_std
        self._n_hidden = n_hidden

        # full_kernel = np.zeros((3, 3, 1, 1))
        # a_kernel = np.array([[-1, -1, -1],
        #                      [-1, 0, -1],
        #                      [-1, -1, -1]])
        # full_kernel[:, :, 0, 0] = a_kernel

        # if self._optimal or not self._optimal:
        # self._kernel = tf.constant(full_kernel, dtype=tf.float32)


        # self._kernel_i_var = tf.Variable(tf.keras.initializers.RandomNormal((range_coloring, range_coloring, 1, n_hidden), stddev = self._weight_std),
        #                                  # stddev=self._weight_std,
        #                                  name="w_kern1"
        #                                  )

        self._kernel_i_var = tf.get_variable(
                                    "w_kern1",
                                    shape=(range_coloring, range_coloring, 1, n_hidden),
                                    initializer = tf.keras.initializers.RandomNormal(stddev = self._weight_std)
                            )

        self._kernel_h_var = tf.get_variable(
                                    "w_kern3",
                                    shape=(range_coloring, range_coloring, 2, n_hidden),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self._weight_std)
                                )

        self._kernel_o_var = tf.get_variable(
                                    "w_kern2",
                                    shape=(1, 1, n_hidden, 2),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self._weight_std)
                                )

        self._kernel_i = self._kernel_i_var
        self._kernel_o = self._kernel_o_var
        self._kernel_h = self._kernel_h_var
        self._bias_i = tf.Variable(tf.zeros([1]))
        self._bias_s = tf.Variable(tf.zeros([1]))


    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def get_params(self):
        if self._optimal:
            return []
        return [self._kernel_i_var, self._kernel_h_var, self._kernel_o_var]
        # return [self._kernel_var, self._deviations]

    def call(self, border, state):
        # hidden, *_ = state
        hidden = state
        f1 = tf.nn.conv2d(
            input=border, filters=self._kernel_i, strides=1, padding="SAME"
        )
        f2 = tf.nn.conv2d(
            input=hidden, filters=self._kernel_h, strides=1, padding="SAME"
        )
        intermediate=tf.math.sigmoid(f1 + f2 + self._bias_i)

        output = tf.math.sigmoid(
            tf.nn.conv2d(
                input=intermediate, filters=self._kernel_o, strides=1, padding="SAME"
            )
            + self._bias_s
        )

        return output, output


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
                     weight_std=getattr(opt.hyper, "init_factor", 1),
                     range_coloring=getattr(opt.hyper, "complex_crossing", 3)
                     )
    if not optimal:
        parameters = fc.get_params()
    else:
        parameters = []

    n_t = opt.dnn.n_t

    data = tf.reshape(data, [-1, data.shape[1], data.shape[2], 1])

    activations = []

    initial_state1 = np.zeros(data.shape[1:3])
    initial_state1[0, :] = 1
    initial_state1[data.shape[1] - 1, :] = 1
    initial_state1[:, 0] = 1
    initial_state1[:, data.shape[2] - 1] = 1

    initial_state2 = np.ones(data.shape[1:3])
    initial_state2[0, :] = 0
    initial_state2[data.shape[1]-1, :] = 0
    initial_state2[:, 0] = 0
    initial_state2[:, data.shape[2] - 1] = 0

    initial_state = np.stack((initial_state1, initial_state2), axis=-1)

    state = tf.constant(initial_state[None, :, :, :], dtype=np.float32)
    with tf.variable_scope("FilledCell") as scope:
        # out = []
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()
            t_output, state = fc(data, state)
            activations.append(state)
            out = t_output
            # out.append(tf.concat([state, t_output], 3))

    if opt.dnn.train_per_step:
        return out, parameters, activations
    else:
        return out, parameters, activations