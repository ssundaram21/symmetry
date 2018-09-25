from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import sys
#import matplotlib
#import matplotlib.pyplot as plt

from pprint import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def activation_function(x):
    return tf.nn.relu(x)

def new_conv_layer(x, stride, weight, bias, padd, activation=True):
    layer = tf.nn.conv2d(input=x,filter=weight, strides=stride, padding=padd)
    layer += bias
    if activation:
        layer = activation_function(layer)
    return layer


'''
1st convolution add 0s at bottom
2nd add 0s at the end(N)
3rd make pading valid
'''
def Crossing(data, opt, dropout_rate, labels_id):

    parameters = []
    activations = []

    C = opt.hyper.complex_crossing

    layer1_padding = tf.constant([[0, 0], [0, 1], [0, 0]])
    data = tf.pad(data, layer1_padding, "CONSTANT")

    data = tf.reshape(data, [-1, opt.dataset.image_size+1, opt.dataset.image_size, 1])
    depth = int(3*C/2)

    print("num neurons crossing: " + str(depth))
    w1 = tf.Variable(tf.truncated_normal([2, 1, 1, 1],
                                            dtype=tf.float32, stddev=opt.hyper.init_factor*1), name='w1')
        #tf.constant(1.0, shape=[2, 1, 1, 1])
    b1 = tf.Variable(0.1*tf.ones([1]), name='b1')
        #tf.constant(-1.0)

    layer1 = new_conv_layer(data, [1, 1, 1, 1], w1, b1, 'SAME')

    activations += [layer1]
    parameters += [w1, b1]

    layer2_padding = tf.constant([[0, 0], [0, 0], [0, opt.dataset.image_size], [0, 0]])
    layer1 = tf.pad(layer1, layer2_padding, "CONSTANT")

    w2 = tf.Variable(tf.truncated_normal([1, opt.dataset.image_size, 1, 1],
                                            dtype=tf.float32, stddev=opt.hyper.init_factor*1), name='w2')
        #tf.constant(1.0, shape=[1, N, 1, depth])
    ''' 
    b2 = []
    for i in range(int(depth/3)):
        for z in range(1, depth+1):
            if z == 3*i + 1:
                b2.append(-(2*i - .5))
            elif z == 3*i + 2:
                b2.append(-2*i)
            elif z == 3*i + 3:
                b2.append(-(2*i+.5))
    '''
    b2 = tf.Variable(0.1 * tf.ones([depth]), name='b2')

    layer2 = new_conv_layer(layer1, [1,1,1,1], w2, b2, 'SAME')

    parameters += [w2, b2]
    activations += [layer2]

    '''
    w3 = []
    for i in range(int(depth/3)):
        for z in range(1, depth+1):
            if z == 3*i + 1:
                w3.append(2.0)
            elif z == 3*i + 2:
                w3.append(-4.0)
            elif z == 3*i + 3:
                w3.append(2.0)
    w3_negative = [-x for x in w3]
    w3 = tf.constant([w3, w3_negative])
    w3 = tf.reshape([tf.transpose(w3)], [1, 1, depth, 2])
    b3 = tf.constant([0.0, 1.0])
    '''
    w3 = tf.Variable(tf.truncated_normal([1, 1, depth, 2],
                                    dtype=tf.float32, stddev=opt.hyper.init_factor*1), name='w3')
    b3 = tf.Variable(0.1 * tf.ones([2]), name='b3')

    layer3 = new_conv_layer(layer2, [1, 1, 1, 1], w3, b3, 'VALID', activation=False)

    layer3 = tf.reshape(layer3, [-1, opt.dataset.image_size+1, opt.dataset.image_size*2, 2])
    layer3 = tf.image.resize_image_with_crop_or_pad(layer3, opt.dataset.image_size, opt.dataset.image_size)
    layer3 = tf.reshape(layer3, [-1, opt.dataset.image_size, opt.dataset.image_size, 2])

    parameters += [w3, b3]
    activations += [layer3]

    return layer3, parameters, activations

    
##
##X=np.array(
##  [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
##  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
##
##sess = tf.InteractiveSession()
##X = tf.cast(X,tf.float32)
##np.set_printoptions(linewidth=400)            
##    
##img = generate_data(20,100,100,20,15)
##plt.imshow(img)
##plt.show()
##img = tf.cast(img,tf.float32)
##
##b = typeA_test(img)
##b = b.eval()
##plt.imshow(b)
##plt.show()



