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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_width = 100
image_height = 100

C = 100
N = image_width

def activation_function(x):
    return tf.nn.relu(x)

def new_conv_layer(x, stride, weight, bias, padd):
    layer = tf.nn.conv2d(input=x,filter=weight, strides=stride, padding=padd)
    layer += bias
    layer=activation_function(layer)
    return layer

'''
1st convolution add 0s at bottom
2nd add 0s at the end(N)
3rd make pading valid
'''
def Crossing(data, opt, dropout_rate, labels_id):


    parameters = []
    activations = []

    layer1_padding = tf.constant([[0, 0], [0, 1], [0, 0]])
    data = tf.pad(data, layer1_padding, "CONSTANT")

    data = tf.reshape(data,[-1,image_width+1,image_height,1])
    depth = int(3*C/2)
    print(depth)
    w1 = tf.constant(1.0, shape=[2, 1, 1, 1])
    b1 = tf.constant(-1.0)

    layer1 = new_conv_layer(data, [1, 1, 1, 1], w1, b1, 'SAME')

    activations += [layer1]
    parameters += [w1, b1]

    layer2_padding = tf.constant([[0, 0], [0, 0], [0, N], [0, 0]])
    layer1 = tf.pad(layer1, layer2_padding, "CONSTANT")

    w2 = tf.constant(1.0,shape=[1, N, 1, depth])
    b2 = []
    for i in range(int(depth/3)):
        for z in range(1, depth+1):
            if z == 3*i + 1:
                b2.append(-(2*i - .5))
            elif z == 3*i + 2:
                b2.append(-2*i)
            elif z == 3*i + 3:
                b2.append(-(2*i+.5))
    b2 = tf.constant(b2,shape=[depth])
    #b2 = tf.reshape(b2,[1,1,1,depth])

    layer2 = new_conv_layer(layer1,[1,1,1,1],w2,b2,'SAME')

    parameters += [w2, b2]
    activations += [layer2]

    w3 = []
    for i in range(int(depth/3)):
        for z in range(1, depth+1):
            if z == 3*i + 1:
                w3.append(2.0)
            elif z == 3*i + 2:
                w3.append(-4.0)
            elif z == 3*i + 3:
                w3.append(2.0)
    w3_negative = [-x  for x in w3]
    w3 = tf.constant([w3, w3_negative])
    w3 = tf.reshape([tf.transpose(w3)], [1, 1, depth, 2])
    b3 = tf.constant([0.0, 1.0])

    layer3 = new_conv_layer(layer2, [1, 1, 1, 1], w3, b3, 'VALID')
    print(layer3)
    layer3 = tf.reshape(layer3, [-1, image_width+1, image_height*2, 2])
    layer3 = tf.image.resize_image_with_crop_or_pad(layer3, image_width, image_height)
    layer3 = tf.reshape(layer3, [-1, image_width, image_height, 2])

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



