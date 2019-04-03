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
                           padding=padding, rate=[dilation_rate, dilation_rate])
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
            h = tf.image.resize_image_with_crop_or_pad(h, tf.shape(skip)[1],  tf.shape(skip)[1])
            h = tf.concat([skip, h], axis=-1)
        for i in range(n):
            h = conv2d(h, num_channels, k_size=3, stride=1, name='de_cconv_{}'.format(i+1))
            h = relu(h, name='relu_{}'.format(i+1))
    return h


def encoder(incoming, base_channels, num_poolings, num_convolutions_step, name='encoder'):
    with tf.variable_scope(name):
        skips = []
        channels = base_channels
        h = incoming
        for i in range(num_poolings):
            print(channels)
            h, skip = encoder_block(h, channels, n=num_convolutions_step, name='block_{}'.format(i+1))
            skips.append(skip)
            if i != num_poolings:
                channels *= 2

        h = conv2d(h, channels, name='block_4a')
        h = relu(h, name='block_4ar')
        h = conv2d(h, channels*2, name='block_4b')
        h = relu(h, name='block_4br')
        h = conv2d(h, channels, name='block_4c')
        h = relu(h, name='block_4cr')
    return h, skips, channels


def decoder(incoming, base_channels, num_convolutions_step, skips, num_classes=2, name='decoder'):
    with tf.variable_scope(name):
        h = incoming
        channels = base_channels
        for i, skip in reversed(list(enumerate(skips))):
            h = decoder_block(h, channels, n=num_convolutions_step,
                              skip=skip, name='de_block_{}'.format(i+1))
            channels = int(channels / 2)

        h = conv2d(h, num_classes, k_size=1, stride=1, name='last_conv')
    return h


def U_net(data, opt, dropout_rate, labels_id):

    data = tf.reshape(data, [-1, opt.dataset.image_size, opt.dataset.image_size, 1])
    data = tf.image.resize_image_with_crop_or_pad(data, opt.dataset.image_size, opt.dataset.image_size)

    base_channels = opt.dnn.base_channels
    num_poolings = opt.dnn.num_poolings
    num_convolutions_step = opt.dnn.num_convolutions_step

    net, skips, last_channels = encoder(data, base_channels, num_poolings, num_convolutions_step, name='encoder')
    predictions = decoder(net, last_channels, num_convolutions_step, skips)

    predictions = tf.image.resize_image_with_crop_or_pad(predictions, opt.dataset.image_size, opt.dataset.image_size)

    return predictions, [], predictions
