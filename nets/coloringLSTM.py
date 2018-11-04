from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell
from tensorflow.python.ops import variable_scope as vs


from pprint import pprint



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

    n_t = getattr(opt.dnn, "n_t", 10)

    data = tf.reshape(data,
                      [-1, opt.dataset.image_size, opt.dataset.image_size, 1])
    
    cell = Conv2DLSTMCell(input_shape=data.shape.dims[-3:],
                          kernel_shape=[3,3],
                          output_channels=getattr(opt.dnn, "layers", 2))    

    state = cell.zero_state(opt.hyper.batch_size, dtype=tf.float32)
    
    with tf.variable_scope("scp") as scope:
        for i in range(n_t):
            if i>0:
                scope.reuse_variables()
            t_output, state = cell(data, state)


    return t_output[:, :, :, :2], cell.weights, state