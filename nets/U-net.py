'''
Basic implementation of U-Net (https://arxiv.org/abs/1505.04597):
    * Encoder blocks: conv + relu + conv + relu + max_pool
    * Decoder blocks: transpose_conv + concat + conv + relu + conv + relu
Optional skip-connections between decoder and the corresponding encoder layer.

Implemented by Amineh Ahmadinejad, inspired by: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/Encoder_Decoder.py
'''


import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(incoming, n_filters, k_size=3, stride=1, padding='SAME', name="conv2d", dilation_rate=1, scope="conv2d"):
    with tf.name_scope(name):
        conv = slim.conv2d(incoming, n_filters, [k_size, k_size], stride,
                           padding=padding, rate=[dilation_rate, dilation_rate], scope=scope)
    return conv


def relu(incoming, name='relu'):
    with tf.name_scope(name):
        output = tf.nn.relu(incoming)
    return output


def transpose_conv2d(incoming, n_filters, k_size=2, stride=2, padding='same', name='trans_conv'):
    with tf.name_scope(name):
        conv = slim.conv2d_transpose(incoming, n_filters, [k_size, k_size], [stride, stride], activation_fn=None, padding=padding)
    return conv


def max_pool(incoming, n):
    return tf.nn.max_pool(incoming, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def encoder_block(incoming, num_channels, name, n=2):
    with tf.variable_scope(name):
        h = incoming
        for i in range(n):
            h = conv2d(h, num_channels, k_size=3, stride=1, name='conv_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        skip = h
        h = max_pool(h, 2)
    return h, skip


def decoder_block(incoming, num_channels, n, skip, has_skip=1, name=''):
    with tf.variable_scope(name):
        h = transpose_conv2d(incoming, num_channels, k_size=2, stride=2)
        if has_skip:
            h = tf.concat([skip, h], axis=-1)
        for i in range(n):
            h = conv2d(h, num_channels, k_size=3, stride=1, name='conv_{}'.format(i+1))
            h = relu(h, name='relu_{}'.format(i+1))
    return h


def encoder(incoming, name='encoder'):
    with tf.variable_scope(name):
        h, skip_1 = encoder_block(incoming, 32, n=2, name='block_1')
        h, skip_2 = encoder_block(h, 64, n=2, name='block_2')
        h, skip_3 = encoder_block(h, 128, n=2, name='block_3')

        h = conv2d(h, 128, name='block_4')
        h = relu(h, name='block_4')
        h = conv2d(h, 256, name='block_4')
        h = relu(h, name='block_4')
        h = conv2d(h, 128, name='block_4')
        h = relu(h, name='block_4')
    return h, [skip_3, skip_2, skip_1]


def decoder(incoming, skips, has_skip=1, num_classes=3, name='decoder'):
    with tf.variable_scope(name):
        h = decoder_block(incoming, 128, n=2, skip=skips[0], has_skip=has_skip, name='block_3')
        h = decoder_block(h, 64, n=2, skip=skips[1], has_skip=has_skip, name='block_2')
        h = decoder_block(h, 32, n=2, skip=skips[2], has_skip=has_skip, name='block_1')
        h = conv2d(h, num_classes, 3, stride=1, name='last_conv')
    return h


def U_net(data, opt, dropout_rate, labels_id):

    data = tf.reshape(data, [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    # TODO: define in opt:
    # has_skip: a boolean for indicating skip connections
    # num_classes: number of labels for each output pixel (default is 3)

    has_skip = opt.has_skip
    num_classes = opt.num_classes

    net, skip = encoder(data, name='encoder')
    predictions = decoder(net, skip, has_skip, num_classes)

    return predictions
