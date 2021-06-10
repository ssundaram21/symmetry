from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell
from tensorflow.python.ops import variable_scope as vs


from pprint import pprint



def LSTM3(data, opt, dropout_rate, labels_id):
    """
    3-Stacked LSTM

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
    iistate1 = tf.constant(istate1, dtype=np.float32)
    state1 = tf.nn.rnn_cell.LSTMStateTuple(istate1_zero, iistate1)


    cell2 = Conv2DLSTMCell(input_shape=[opt.dataset.image_size, opt.dataset.image_size, 64],
                          kernel_shape=[3, 3],
                          output_channels=64, name='b')
    istate2_zero = tf.constant(np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64]),
                               dtype=np.float32)
    istate2 = np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64])
    iistate2 = tf.constant(istate2, dtype=np.float32)
    state2 = tf.nn.rnn_cell.LSTMStateTuple(istate2_zero, iistate2)

    cell3 = Conv2DLSTMCell(input_shape=[opt.dataset.image_size, opt.dataset.image_size, 64],
                           kernel_shape=[3, 3],
                           output_channels=64, name='c')

    istate3_zero = tf.constant(np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64]),
                               dtype=np.float32)
    istate3 = np.zeros([opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size, 64])
    iistate3 = tf.constant(istate3, dtype=np.float32)
    state3 = tf.nn.rnn_cell.LSTMStateTuple(istate3_zero, iistate3)


    out = []
    act_state1 = []
    act_state2 = []
    act_state3 = []
    act_fc = []
    out_1 = []
    out_2 = []
    out_3 = []
    # weights1 = tf.get_variable('W1', shape=(1, 64), initializer = tf.compat.v1.initializers.glorot_uniform())
    # bias1 = tf.get_variable('B1', shape=(64), initializer = tf.compat.v1.initializers.glorot_uniform())
    # weights2 = tf.get_variable('W2', shape=(1, 512), initializer=tf.compat.v1.initializers.glorot_uniform())
    # bias2 = tf.get_variable('B2', shape=(512), initializer=tf.compat.v1.initializers.glorot_uniform())

    with tf.variable_scope("scp") as scope:
        for i in range(n_t):
            if i > 0:
                scope.reuse_variables()

            tt1, state1 = cell1(data, state1)
            tt2, state2 = cell2(tt1[:, :, :, :64], state2)
            t_output, state3 = cell3(tt2[:, :, :, :64], state3)

            out.append([t_output[:, :, :, :64]])

            out_1.append(tt1)
            out_2.append(tt2)
            out_3.append(t_output)
            act_state1.append(state1)
            act_state2.append(state2)
            act_state3.append(state3)

        # ### Global average pooling layer
        # print("OUT SHAPE:", out[-1][0].shape)
        # pool_out = tf.nn.avg_pool(out[-1][0], ksize = [1, 20, 20, 1], strides = [1, 20, 20, 1], padding="SAME")
        # print("POOL OUTPUT SHAPE:", pool_out.shape)
        # # [32, 2]

        # add two ReLu fully connected layers
        output = out[-1][0]
        flat = tf.reshape(output, [-1, 64*400])
        print("\n\nFLAT SHAPE:", flat.shape)

    with tf.variable_scope("fully_connected") as scope:
        fc1_out = tf.contrib.layers.fully_connected(flat, num_outputs=512, activation_fn=tf.nn.relu)
        print("\n\nFC1 OUTPUT SHAPE:", fc1_out.shape)
        act_fc.append(fc1_out)
        fc2_out = tf.contrib.layers.fully_connected(fc1_out, num_outputs=2, activation_fn=None)
        print("\n\nFC2 OUTPUT SHAPE:", fc2_out.shape)

    if opt.dnn.train_per_step:
        return fc2_out, [cell1.weights, cell2.weights], [act_state1, act_state2, act_state3, act_fc]
    else:
        return fc2_out, [], [act_state1, act_state2, act_state3, out_1, out_2, out_3, act_fc]
