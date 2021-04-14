import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import ConvLSTM2D, RNN, Layer

class RNNCell(Layer):
    """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

    def __init__(
        self,
        input_shape,
        n_filters,
        kernel_i=None,
        kernel_h=None,
        kernel_o=None,
        biases=None,
        init_scale=None,
    ):

        super().__init__(input_shape=input_shape, name="rnn_cell")

        self.data_format = "channels_last"

        self.init_scale = init_scale or 1.0

        self.n_filters = n_filters
        self.state_size = tf.TensorShape(input_shape)

        def get_init(k, sc):
            if k is None:
                return tf.keras.initializers.RandomNormal(stddev=sc)
            else:
                return tf.keras.initializers.constant(k)

        self.kernel_i = self.add_weight(
            shape=(3, 3, 1, self.n_filters),
            name="kernel_i",
            initializer=get_init(kernel_i, self.init_scale),
        )
        self.kernel_h = self.add_weight(
            shape=(3, 3, 1, self.n_filters),
            name="kernel_h",
            initializer=get_init(kernel_h, self.init_scale),
        )
        self.kernel_o = self.add_weight(
            shape=(1, 1, self.n_filters, 1),
            name="kernel_o",
            initializer=get_init(kernel_o, self.init_scale),
        )
        self.biases = self.add_weight(
            shape=(self.n_filters + 1,),
            name="biases",
            initializer=tf.keras.initializers.zeros()
            if biases is None
            else tf.keras.initializers.constant(biases),
        )

    def build(self, input_shape):
        super().build(input_shape)

    @property
    def output_size(self):
        return self.input_shape

    def compute_output_shape(self, input_shape):
        return self.input_shape

    def call(self, inputs, state):
        hidden, *_ = state
        f1 = tf.nn.conv2d(
            input=inputs, filters=self.kernel_i, strides=1, padding="SAME"
        )
        f2 = tf.nn.conv2d(
            input=hidden, filters=self.kernel_h, strides=1, padding="SAME"
        )
        intermediate = tf.math.sigmoid(f1 + f2 + self.biases[0 : self.n_filters])

        output = tf.math.sigmoid(
            tf.nn.conv2d(
                input=intermediate, filters=self.kernel_o, strides=1, padding="SAME"
            )
            + self.biases[self.n_filters]
        )
        return output, output


def MultiLSTM(data, opt, dropout_rate, labels_id):
    train_input = data
    imsize = opt.dataset.image_size
    n_steps = opt.dnn.n_t
    n_filters = 5
    init_std = 0
    return_sequences=False
    print(data.shape.dims[-3:]+[1])
    cell_rand = RNNCell(
        input_shape=data.shape.dims[-3:]+[1], n_filters=n_filters, init_scale=init_std
    )
    conv_rand = RNN(cell=cell_rand, unroll=True, return_sequences=return_sequences)

    train_output = conv_rand(
        tf.tile(train_input[:, None, :, :, :1], (1, n_steps, 1, 1, 1)),
        initial_state=(train_input[:, :, :, 1:],),
    )

    return train_output


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib.rnn import Conv2DLSTMCell
# from tensorflow.python.ops import variable_scope as vs
#
#
# from pprint import pprint
#
#
#
# def MultiLSTM(data, opt, dropout_rate, labels_id):
#     """ Run the coloring network on data, with hyperparameters
#
#     :param data: in the shape batch_size, image_height, image_width, 2
#         where the last two channels are the inside and outside contour
#     :param opt:
#         opt.hyper parameters are
#         n_t number of timepoints for the network to run
#         n_hidden number of hidden layers
#     :param dropout_rate:
#     :param labels_id:
#     :return:
#     """
#
#     n_t = getattr(opt.dnn, "n_t", 10)
#
#     data = tf.reshape(data,
#                       [-1, opt.dataset.image_size, opt.dataset.image_size, 1])
#
#     print(data.shape.dims[-3:])
#     cell1 = Conv2DLSTMCell(input_shape=data.shape.dims[-3:],
#                           kernel_shape=[3, 3],
#                           output_channels=64, name='a')
#     state1 = cell1.zero_state(opt.hyper.batch_size, dtype=tf.float32)
#
#
#     cell2 = Conv2DLSTMCell(input_shape=[opt.dataset.image_size, opt.dataset.image_size, 64],
#                           kernel_shape=[1, 1],
#                           output_channels=2, name='b')
#     state2 = cell2.zero_state(opt.hyper.batch_size, dtype=tf.float32)
#
#
#     out = []
#     act_state1 = []
#     act_state2 = []
#     with tf.variable_scope("scp") as scope:
#         for i in range(n_t):
#             if i > 0:
#                 scope.reuse_variables()
#
#             tt, state1 = cell1(data, state1)
#             t_output, state2 = cell2(tt[:, :, :, :64], state2)
#
#             out.append([t_output[:, :, :, :2]])
#             act_state1.append(state1)
#             act_state2.append(state2)
#
#     if opt.dnn.train_per_step:
#         return out, [cell1.weights, cell2.weights], [act_state1, act_state2]
#     else:
#         return out[-1], [], [act_state1, act_state2]
