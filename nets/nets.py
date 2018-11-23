import tensorflow as tf


from nets.alexnet import Alexnet as net_Alexnet
from nets.MLP import MLP1 as net_MLP1
from util import summary as summ

from nets.crossing import Crossing as net_Crossing
from nets.crossing import Crossing_Pertubration as net_Crossing_Pertubration

from nets.crossing_learning import Crossing_Perturbation as net_Crossing_Learning_Pertubration
from nets.crossing_learning import Crossing as net_Crossing_Learning

from nets.coloring import Coloring as net_Coloring
from nets.coloringLSTM import ColoringLSTM as net_ColoringLSTM

def MLP1(x, opt, dropout_rate, labels_id):
    return net_MLP1(x, opt, dropout_rate, labels_id)

def Crossing(x, opt, dropout_rate, labels_id):
    return net_Crossing(x, opt, dropout_rate, labels_id)

def Crossing_Learning(x, opt, dropout_rate, labels_id):
    return net_Crossing_Learning(x, opt, dropout_rate, labels_id)


def Crossing_Perturbation(x, opt, delta, labels_id):
    return net_Crossing_Pertubration(x, opt, delta, labels_id)

def Crossing_Learning_Perturbation(x, opt, delta, labels_id):
    return net_Crossing_Learning_Pertubration(x, opt, delta, labels_id)


def Coloring(x, opt, dropout_rate, labels_id):
    return net_Coloring(x, opt, dropout_rate, labels_id)


def ColoringLSTM(x, opt, dropout_rate, labels_id):
    return net_ColoringLSTM(x, opt, dropout_rate, labels_id)