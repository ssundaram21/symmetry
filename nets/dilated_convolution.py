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
    with tf.variable_scope(name):
        h = incoming
        for i in range(len(channels)):
            h = conv2d(h, channels[i], dilation_rate=dilations[i], scope='dilated_conv2d_%d' % (i+1))
            h = relu(h, name='relu_{}'.format(i+1))
    return h


def Dilated_convolution(data, opt, dropout_rate, labels_id):

    data = tf.reshape(data, [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    # TODO: add channels and dilations to opt
    ''' they should look something like this: 
        channels = [3, 3, 3, 3, 3]
        dilations = [1, 2, 4, 1, 1] 
        (channels[-1] must equal 3 to match the number of classes of output (inside, outside, border).
        '''

    channels = opt.channels
    dilations = opt.dilations

    predictions = model(data, channels, dilations)
    return predictions
