import tensorflow as tf


from nets.alexnet import Alexnet as net_Alexnet
from nets.MLP import MLP1 as net_MLP1
from util import summary as summ


def MLP1(x, opt, dropout_rate, labels_id):
    return net_MLP1(x, opt, dropout_rate, labels_id)

