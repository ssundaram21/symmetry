from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell
from tensorflow.python.ops import variable_scope as vs

from pprint import pprint

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def ColoringLSTM(data, opt, dropout_rate, labels_id):
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

    n_t = getattr(opt.dnn, "n_t", 30)
    print(getattr(opt.dnn, "layers", 2))
    cell = Conv2DLSTMCell(input_shape=data.shape[-3:],
                          kernel_shape=[opt.hyper.complex_crossing, opt.hyper.complex_crossing],
                          output_channels=getattr(opt.dnn, "layers", 2))

    data = tf.reshape(data,
                      [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    inp_time = tf.tile(data[:, None, :, :, :], [1, n_t, 1, 1, 1])

    (outputs, stat) = tf.nn.dynamic_rnn(cell, inp_time, time_major=False,
                                        dtype=tf.float32)

    return outputs[:, n_t - 1, :, :, :2], [vs.get_variable("kernel")], stat



