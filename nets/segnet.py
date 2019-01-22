'''
Encoder-decoder approach for semantic segmentation, based on SegNet [https://arxiv.org/pdf/1511.00561.pdf].

SegNet properties:
    * uses memorized pooling indices for up-sampling
    * no skip connections
    * Encoder:
        [conv + batch_normalization + relu] + [conv + batch_normalization + relu] + pooling
    * Decoder:
        Up-sampling + [conv + batch_normalization + relu] + [conv + batch_normalization + relu]

Implemented by Amineh Ahmadinejad, inspired by https://github.com/aizawan/segnet/blob/master/segnet.py
'''


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages


def conv2d(incoming, n_filters, k_size=3, stride=1, padding='SAME', name="conv2d", dilation_rate=1, scope="conv2d"):
    with tf.name_scope(name):
        conv = slim.conv2d(incoming, n_filters, [k_size, k_size], stride,
                           padding=padding, rate=[dilation_rate, dilation_rate], scope=scope)
    return conv


def relu(incoming, name='relu'):
    with tf.name_scope(name):
        output = tf.nn.relu(incoming)
    return output


# copy right: https://github.com/aizawan/segnet/blob/master/ops.py
def batch_norm(incoming, training,
               epsilon = 1e-4, alpha = 0.1, decay = 0.9,
               beta_init = tf.constant_initializer(0.0),
               gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
               reuse = False, name = 'batch_norm'):

    input_shape = incoming.get_shape().as_list()
    depth = input_shape[-1]
    with tf.name_scope(name) as scope:
        beta = tf.get_variable(name + '_beta', shape=depth,
                               initializer=beta_init, trainable=True)
        gamma = tf.get_variable(name + '_gamma', shape=depth,
                                initializer=gamma_init, trainable=True)

        axes = list(range(len(input_shape) - 1))
        batch_mean, batch_variance = tf.nn.moments(incoming, axes) # channel
        moving_mean = tf.get_variable(
            name + '_moving_mean', shape=depth,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable(
            name + '_moving_variance', shape=depth,
            initializer=tf.constant_initializer(1.0),
            trainable=False)

        def update():
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, batch_mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, batch_variance, decay, zero_debias=False)

            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = tf.cond(training, update, lambda: (moving_mean, moving_variance))

        output = tf.nn.batch_normalization(incoming, mean, variance, beta, gamma, epsilon)
    return output


def maxpool2d_with_argmax(incoming, pool_size=2, stride=2,
                          name='maxpool_with_argmax'):
    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        pooled, mask = tf.nn.max_pool_with_argmax(x, ksize=filter_shape, strides=strides, padding='SAME')
    return pooled, mask


def maxunpool2d(incoming, mask, stride=2, name='unpool'):
    strides = tf.constant([1, stride, stride, 1])
    output_shape = tf.multiply(tf.shape(incoming), strides)
    output_shape_flat = [tf.reduce_prod(output_shape), 1]

    with tf.name_scope(name):
        input_flat = tf.reshape(incoming, [-1, 1])
        mask_flat = tf.cast(tf.reshape(mask, [-1, 1]), tf.int32)
        upsampled_flat = tf.scatter_nd(mask_flat, input_flat, output_shape_flat, 'constant')
        upsampled = tf.reshape(upsampled_flat, output_shape)
        return upsampled


def encoder_block(incoming, training, n, num_channels, name):
    with tf.variable_scope(name):
        h = incoming
        for i in range(n):
            h = conv2d(h, num_channels, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, training, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        h, mask = maxpool2d_with_argmax(h, name='maxpool_{}'.format(n))
    return h, mask


def decoder_block(incoming, mask, training, n, num_channels, name='decoder', adj_k=True):
    with tf.variable_scope(name):
        h = maxunpool2d(incoming, mask, 2, name='unpool')
        input_shape = incoming.get_shape().as_list()
        h = tf.reshape(h, shape=[-1, input_shape[1]*2, input_shape[2]*2, input_shape[3]])
        for i in range(n):
            if i == (n - 1) and adj_k:
                h = conv2d(h, num_channels / 2, 3, stride=1, name='conv_{}'.format(i + 1))
            else:
                h = conv2d(h, num_channels, 3, stride=1, name='conv_{}'.format(i + 1))
        h = batch_norm(h, training, name='bn_{}'.format(n))
        h = relu(h, name='relu_{}'.format(n))
    return h


def encoder(incoming, training, name='encoder'):
    with tf.variable_scope(name):
        h, mask_1 = encoder_block(incoming, training, n=3, num_channels=32, name='block_1')
        h, mask_2 = encoder_block(h, training, n=3, num_channels=64, name='block_2')
        h, mask_3 = encoder_block(h, training, n=3, num_channels=128, name='block_3')
    return h, [mask_3, mask_2, mask_1]


def decoder(incoming, mask, training, num_classes=3, name='decoder'):
    with tf.variable_scope(name):
        h = decoder_block(incoming, mask[0], training, n=3, num_channels=128, name='block_3')
        h = decoder_block(h, mask[1], training, n=3, num_channels=64, name='block_2')
        h = decoder_block(h, mask[2], training, n=3, num_channels=32, name='block_1')
        h = conv2d(h, num_classes, 3, stride=1, name='last_conv')
    return h


def Segnet(data, opt, dropout_rate, labels_id):

    data = tf.reshape(data, [-1, opt.dataset.image_size, opt.dataset.image_size, 1])

    # TODO: define in opt:
    # training: a boolean indicating weather or not we are in training phase. this is for batch_norm in encoder
    # num_classes: number of labels for each output pixel (default is 3).

    training = opt.training
    num_classes = opt.num_classes

    net, mask = encoder(data, training, name='encoder')
    predictions = decoder(net, mask, training, num_classes, name='decoder')

    return predictions