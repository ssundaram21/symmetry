import tensorflow as tf
from util import summary as summ
from numpy import *



def MLP3(x, opt, labels_id, dropout_rate):


    parameters = []

    aa = x
    num_neurons_before_fc = int(prod(aa.get_shape()[1:]))
    flattened = tf.reshape(aa, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(12 * opt.dnn.neuron_multiplier[0])],
                                            dtype=tf.float32,stddev=1e-3), name='weights')
        b = tf.Variable(0.1*tf.ones([int(512 * opt.dnn.neuron_multiplier[0])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))
        summ.activation_summaries(fc1, opt)
        dropout1 = tf.nn.dropout(fc1, dropout_rate)

    # fc2
    with tf.name_scope('fc2') as scope:
        W = tf.Variable(tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[0]), int(512 * opt.dnn.neuron_multiplier[1])],
                                            dtype=tf.float32,stddev=1e-3), name='weights')
        b = tf.Variable(0.1*tf.ones([int(512 * opt.dnn.neuron_multiplier[1])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope))
        summ.activation_summaries(fc2, opt)
        dropout2 = tf.nn.dropout(fc2, dropout_rate)

    # fc3
    with tf.name_scope('fc3') as scope:
        W = tf.Variable(tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[1]), int(512 * opt.dnn.neuron_multiplier[2])],
                                            dtype=tf.float32,stddev=1e-3), name='weights')
        b = tf.Variable(0.1*tf.ones([int(512 * opt.dnn.neuron_multiplier[2])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout2, W), b, name=scope))
        summ.activation_summaries(fc3, opt)
        dropout3 = tf.nn.dropout(fc3, dropout_rate)


    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[2]), len(labels_id)],
                                         dtype=tf.float32,
                                         stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([len(labels_id)]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout3, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters


def MLP1(x, opt, labels_id, dropout_rate):
    parameters = []

    aa = x
    num_neurons_before_fc = int(prod(aa.get_shape()[1:]))
    flattened = tf.reshape(aa, [-1, num_neurons_before_fc])

    # fc1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([num_neurons_before_fc, int(512 * opt.dnn.neuron_multiplier[0])], dtype=tf.float32, stddev=1e-3),
                        name='weights')
        b = tf.Variable(0.1 * tf.ones([int(512 * opt.dnn.neuron_multiplier[0])]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope))
        summ.activation_summaries(fc1, opt)
        dropout1 = tf.nn.dropout(fc1, dropout_rate)


    # fc8
    with tf.name_scope('fc_out') as scope:
        W = tf.Variable(tf.truncated_normal([int(512 * opt.dnn.neuron_multiplier[0]), len(labels_id)],
                                            dtype=tf.float32,
                                            stddev=1e-2), name='weights')
        b = tf.Variable(tf.zeros([len(labels_id)]), name='bias')
        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout1, W), b, name=scope)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters

