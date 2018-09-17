import tensorflow as tf


from nets.alexnet import Alexnet as net_Alexnet
from util import summary as summ


def Alexnet(x, dropout_rate, opt, labels_id):
    return net_Alexnet(x, opt, labels_id, dropout_rate)

