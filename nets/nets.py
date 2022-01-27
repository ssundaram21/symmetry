from nets.dilated_convolution import Dilated_convolution as net_Dilated_convolution
from nets.LSTM3 import LSTM3 as net_LSTM3

def LSTM3(data, opt, dropout_rate, labels_id):
    return net_LSTM3(data, opt, dropout_rate, labels_id)

def Dilation(x, opt, dropout_rate, labels_id):
    return net_Dilated_convolution(x, opt, dropout_rate, labels_id)