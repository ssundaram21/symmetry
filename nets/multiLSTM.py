from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell
from tensorflow.python.ops import variable_scope as vs


from pprint import pprint



def MultiLSTM(data, opt, dropout_rate, labels_id):
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

    print(data.shape.dims[-3:])
    cell1 = Conv2DLSTMCell(input_shape=data.shape.dims[-3:],
                          kernel_shape=[3, 3],
                          output_channels=64, name='a')
    state1 = cell1.zero_state(opt.hyper.batch_size, dtype=tf.float32)

    cell2 = Conv2DLSTMCell(input_shape=[opt.dataset.image_size, opt.dataset.image_size, 64],
                          kernel_shape=[1, 1],
                          output_channels=2, name='b')
    state2 = cell2.zero_state(opt.hyper.batch_size, dtype=tf.float32)

    out = []
    with tf.variable_scope("scp") as scope:
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()

            tt, state1 = cell1(data, state1)
            t_output, state2 = cell2(tt[:, :, :, :64], state2)

            out.append([t_output[:, :, :, :2]])

    if opt.dnn.train_per_step:
        return out, [cell1.weights, cell2.weights], [state1, state2]
    else:
        return out[-1], [], [state1]
