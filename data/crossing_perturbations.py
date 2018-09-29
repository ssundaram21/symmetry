from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import sys
import random as rnd
#import matplotlib
#import matplotlib.pyplot as plt

from pprint import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_width = 30
image_height = 30

C = 6
N = image_width

def activation_function(x):
    return tf.nn.relu(x)

def new_conv_layer(x, stride, weight, bias, padd):
    layer = tf.nn.conv2d(input=x,filter=weight, strides=stride, padding=padd)
    layer += bias
    layer = activation_function(layer)
    return layer

def variable(init, delta, shape=(1,)):
    start = rnd.gauss(init, delta)
    initial = tf.constant(start, shape=shape)
    return tf.Variable(initial)

def constant(init, delta=0, shape=(1,)):
    if delta > 0:
        start = rnd.gauss(init, delta)
    else:
        start = init
    return tf.constant(start, shape=shape)

    
def crossing_test(data, delta=0, fixed_weights=True, training=False):
    layer1_padding = tf.constant([[0,0],[0,1],[0,0]])
    data = tf.pad(data, layer1_padding, "CONSTANT")
    
    data = tf.reshape(data,[-1,image_width+1,image_height,1])
    depth = int(3*C/2)
    
    if fixed_weights:
        w1 = constant(1.0,delta,shape=[2,1,1,1])
        b1 = constant(-1.0,delta)
    else:
        w1 = variable(1.0,delta,shape=[2,1,1,1])
        b1 = variable(-1.0,delta)
        
    layer1 = new_conv_layer(data,[1,1,1,1],w1,b1,'SAME')

    layer2_padding = tf.constant([[0,0],[0,0],[0,N],[0,0]])
    layer1 = tf.pad(layer1, layer2_padding, "CONSTANT")

    if fixed_weights:
        w2 = constant(1.0,delta,shape=[1,N,1,depth])
    else:
        w2 = variable(1.0,delta,shape=[1,N,1,depth])
        
    b2 = []
    for i in range(int(depth/3)):
        for z in range(1,depth+1):
            if z == 3*i + 1:
                b2.append(rnd.gauss(-(2*i - .5),delta))
            elif z == 3*i + 2:
                b2.append(rnd.gauss(-2*i,delta))
            elif z == 3*i + 3:
                b2.append(rnd.gauss(-(2*i+.5),delta))

    if fixed_weights:
        b2 = tf.constant(b2,shape=[depth])
    else:
        b2 = tf.Variable(b2,shape=[depth])

    layer2 = new_conv_layer(layer1,[1,1,1,1],w2,b2,'SAME')
    
    w3 = []
    for i in range(int(depth/3)):
        for z in range(1,depth+1):
            if z == 3*i + 1:
                w3.append(rnd.gauss(2.0,delta))
            elif z == 3*i + 2:
                w3.append(rnd.gauss(-4.0,delta))
            elif z == 3*i + 3:
                w3.append(rnd.gauss(2.0,delta))

    if fixed_weights:
        w3 = tf.constant(w3)
        w3 = tf.reshape(w3,[1,1,depth,1])
        b3 = tf.constant(rnd.gauss(0.0,delta))
    else:
        w3 = tf.Variable(w3)
        w3 = tf.reshape(w3,[1,1,depth,1])
        b3 = tf.Variable(rnd.gauss(0.0,delta))

    layer3 = new_conv_layer(layer2,[1,1,1,1],w3,b3,'VALID')
    layer3 = tf.reshape(layer3,[-1,image_width+1,image_height*2,1])
    layer3 = tf.image.resize_image_with_crop_or_pad(layer3, image_width, image_height)
    layer3 = tf.reshape(layer3,[-1,image_width*image_height])
    
    return layer3






