import tensorflow as tf
from util import summary as summ
from numpy import *


import numpy as np

import sys

num_neurons = [96, 256, 384, 192]


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def Alexnet(x, opt, labels_id, dropout_rate):
    reuse = False
    global num_neurons

    STRIDE = opt.dnn.stride



    parameters = []
    activations = []
    # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, 3, int(num_neurons[0])],
            stddev=5e-2 , dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=
                                 tf.constant(0.0, shape=[int(num_neurons[0])]),
                                 name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        print(conv1.shape[1:])

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)
        activations += [conv1]

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3,  1],
                           strides=[1, 1, 1, 1],
                           padding='SAME', name='pool1')

    with tf.name_scope('lrn1') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(pool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, int(num_neurons[0]),
             int(num_neurons[1])],
            stddev=5e-2, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[1])]), name='biases')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        print(conv2.shape[1:])

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv2, opt)
        activations += [conv2]

    # pool2
    if STRIDE == 1:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, opt.dnn.neuron_multiplier[1], opt.dnn.neuron_multiplier[1], 1],
                             strides=[1, STRIDE, STRIDE, 1], padding='VALID', name='pool2')
    else:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, opt.dnn.neuron_multiplier[1], opt.dnn.neuron_multiplier[1], 1],
                             strides=[1, STRIDE, STRIDE, 1], padding='SAME', name='pool2')

    with tf.name_scope('lrn2') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(pool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # local3
    with tf.variable_scope('local3', reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        print(lrn2.shape[1:])

        dim = int(prod(lrn2.get_shape()[1:]))
        pool_vec = tf.reshape(lrn2, [-1, dim])

        nneurons = int(num_neurons[2])
        weights = tf.get_variable(
            initializer=tf.truncated_normal([dim, nneurons],
                                            stddev=0.04 ,
                                            dtype=tf.float32), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[nneurons]), name='biases')

        local3t = tf.nn.relu(tf.matmul(pool_vec, weights) + biases, name=scope.name)
        local3 = tf.nn.dropout(local3t, dropout_rate)

        print(np.prod(local3.shape[1:]))

        activations += [local3]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(local3, opt)

    # local4
    with tf.variable_scope('local4', reuse=reuse) as scope:
        weights = tf.get_variable(
            initializer=tf.truncated_normal([nneurons, int(num_neurons[3])],
                                            stddev=0.04 ,
                                            dtype=tf.float32), name='weights')

        biases = tf.get_variable(
            initializer=tf.constant(0.1, shape=[int(num_neurons[3])]), name='biases')
        local4t = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        local4 = tf.nn.dropout(local4t, dropout_rate)

        print(np.prod(local4.shape[1:]))

        activations += [local4]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(local4, opt)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        weights = tf.get_variable(
            initializer=tf.truncated_normal([int(num_neurons[3]), len(labels_id)],
                                            stddev=1 / (float(num_neurons[3])), dtype=tf.float32),
            name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[len(labels_id)]), name='biases')
        fc8 = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        activations += [fc8]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters, activations
