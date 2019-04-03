from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell
from tensorflow.python.ops import variable_scope as vs


from pprint import pprint



def MultiLSTM_init(data, opt, dropout_rate, labels_id):
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

    #print(data.shape.dims[-3:])
    cell1 = Conv2DLSTMCell(input_shape=data.shape.dims[-3:],
                          kernel_shape=[3, 3],
                          output_channels=64, name='a')

    istate1_zero = tf.constant(np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64]),
                               dtype=np.float32)
    istate1 = np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64])
    istate1[:, 0, :, :] = 1
    istate1[:, data.shape[1] - 1, :, :] = 1
    istate1[:, :, 0, :] = 1
    istate1[:, :, data.shape[2] - 1, :] = 1
    iistate1 = tf.constant(istate1, dtype=np.float32)

    state1 = tf.nn.rnn_cell.LSTMStateTuple(istate1_zero, iistate1)

    cell2 = Conv2DLSTMCell(input_shape=[opt.dataset.image_size, opt.dataset.image_size, 64],
                          kernel_shape=[1, 1],
                          output_channels=2, name='b')

    istate2_zero = tf.constant(np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 2]),
                               dtype=np.float32)
    istate2 = np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 2])
    istate2[:, 0, :, :] = 1
    istate2[:, data.shape[1] - 1, :, :] = 1
    istate2[:, :, 0, :] = 1
    istate2[:, :, data.shape[2] - 1, :] = 1
    iistate2 = tf.constant(istate2, dtype=np.float32)

    state2 = tf.nn.rnn_cell.LSTMStateTuple(istate2_zero, iistate2)


    out = []
    act_state1 = []
    act_state2 = []
    out_1 = []
    out_2 = []
    with tf.variable_scope("scp") as scope:
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()

            tt, state1 = cell1(data, state1)
            t_output, state2 = cell2(tt[:, :, :, :64], state2)
            out.append([t_output[:, :, :, :2]])

            out_1.append(tt)
            out_2.append(t_output)
            act_state1.append(state1)
            act_state2.append(state2)

    if opt.dnn.train_per_step:
        return out, [cell1.weights, cell2.weights], [act_state1, act_state2]
    else:
        return out[-1], [], [act_state1, act_state2, out_1, out_2]
