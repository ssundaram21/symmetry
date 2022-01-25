

from nets.alexnet import Alexnet as net_Alexnet
from nets.MLP import MLP1 as net_MLP1
from util import summary as summ

from nets.crossing import Crossing as net_Crossing
from nets.crossing import Crossing_Pertubration as net_Crossing_Pertubration

from nets.crossing_learning import Crossing_Perturbation as net_Crossing_Learning_Pertubration
from nets.crossing_learning import Crossing as net_Crossing_Learning

from nets.coloring import Coloring as net_Coloring


from nets.coloringLSTM import ColoringLSTM as net_ColoringLSTM
from nets.coloringLSTM_init import ColoringLSTM as net_ColoringLSTM_init


from nets.dilated_convolution import Dilated_convolution as net_Dilated_convolution

from nets.unet import U_net as net_U_net

from nets.segnet import Segnet as net_Segnet

from nets.optimalLSTM import optimalLSTM as net_optimalLSTM
from nets.multiLSTM import MultiLSTM as net_MultiLSTM
from nets.multiLSTM_init import MultiLSTM_init as net_MultiLSTM_init
from nets.LSTM3 import LSTM3 as net_LSTM3

def LSTM3(data, opt, dropout_rate, labels_id):
    return net_LSTM3(data, opt, dropout_rate, labels_id)

def Dilation(x, opt, dropout_rate, labels_id):
    return net_Dilated_convolution(x, opt, dropout_rate, labels_id)