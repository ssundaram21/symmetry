import tensorflow as tf


from nets.alexnet import Alexnet as net_Alexnet
from nets.MLP import MLP1 as net_MLP1
from util import summary as summ
from nets.crossing import Crossing as net_Crossing
from nets.crossing_learning import Crossing as net_Crossing_Learning
from nets.coloring import Coloring as net_Coloring

def MLP1(x, opt, dropout_rate, labels_id):
    return net_MLP1(x, opt, dropout_rate, labels_id)

def Crossing(x, opt, dropout_rate, labels_id):
    return net_Crossing(x, opt, dropout_rate, labels_id)

def Crossing_Learning(x, opt, dropout_rate, labels_id):
    return net_Crossing_Learning(x, opt, dropout_rate, labels_id)

def Coloring(x, opt, dropout_rate, labels_id):
    return net_Coloring(x, opt, dropout_rate, labels_id)