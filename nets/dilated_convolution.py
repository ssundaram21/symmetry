"""
Insideness with semantic segmentation.

Base idea: "MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS" (https://arxiv.org/abs/1511.07122):
    * no encoder-decoder
    * dilated convolutions with different channels

Implemented by Amineh Ahmadinejad
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(incoming, n_filters, k_size=3, stride=1, padding='SAME', name="conv2d", dilation_rate=1, scope="conv2d"):
    with tf.variable_scope(name):
        conv = slim.conv2d(incoming, n_filters, [k_size, k_size], stride,
                           padding=padding, rate=[dilation_rate, dilation_rate], scope=scope)
    return conv


def relu(incoming, name='relu'):
    with tf.name_scope(name):
        output = tf.nn.relu(incoming)
    return output


def model(incoming, channels, dilations, name="model"):
    """
    makes the network.

    :param incoming: a tensor of shape (batch_size, width, height, channel_size) as input data.
    :param channels: a list of length h, where h is number of the number of network layers.
            channels[i] defines the number of convolutional filters at 'i'th layer.
            last element of channels should be the same as number of classes (i.e. 3).
    :param dilations: a list with the same length as channels.
             dilations[i] is the dilation rate applied to 'i'th layer.
    :param name: variable_scope name
    :return: output of network. A tensor of shape (batch_size, width, height, channels[-1])
    """

    act = []
    with tf.variable_scope(name):
        h = incoming
        for i in range(len(channels)):
            if i == len(channels)-1:
                h = conv2d(h, channels[i], k_size=1, dilation_rate=dilations[i], scope='dilated_conv2d_%d' % (i+1))
            else:
                h = conv2d(h, channels[i], dilation_rate=dilations[i], scope='dilated_conv2d_%d' % (i+1))
            h = relu(h, name='relu_{}'.format(i+1))
            act.append(h)

    return h, act


def Dilated_convolution(data, opt, dropout_rate, labels_id):

    data = tf.reshape(data, [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    channel_rate = opt.dnn.complex_dilation
    num_layers = opt.dnn.num_layers
    no_dilation = opt.dnn.no_dilation

    channels = [2*channel_rate] + [(2**i) * channel_rate for i in range(1, num_layers-1)] + [2]
    if no_dilation:
        dilations = [1 for _ in range(num_layers)]
    else:
        dilations = [1] + [(2**i) for i in range(num_layers-3)] + [1, 1]

    predictions, activations = model(data, channels, dilations)
    return predictions, [], activations
