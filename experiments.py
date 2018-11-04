import numpy as np
import sys
import datasets
import copy

import pickle

class DNN(object):

    def __init__(self):
        self.name = "MLP1"
        self.pretrained = False
        self.version = 1
        self.layers = 4
        self.stride = 2
        self.neuron_multiplier = np.ones([self.layers])
        self.n_t = 1

    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 1#0.95
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.alpha = 0.1
        self.augmentation = False


class Experiments(object):

    def __init__(self, id, name, dataset, output_path, family_id, family_name):
        self.name = "base"
        self.log_dir_base = output_path

        # Recordings
        self.max_to_keep_checkpoints = 2

        # Test after training:
        self.skip_train = False

        # Start from scratch even if it existed
        self.restart = True

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        self.family_ID = family_id
        self.family_name = family_name

        # Add additional descriptors to Experiments
        self.dataset = dataset
        self.dnn = DNN()
        self.hyper = Hyperparameters()


def generate_experiments_dataset(opt_data):
    return Experiments(opt_data.ID, opt_data.name, opt_data, opt_data.log_dir_base, 0, 'data')


def change_dataset(opt, opt_data):
    opt.dataset = opt_data


def get_experiments(output_path):

    opt_data = datasets.get_datasets(output_path)

    # # #
    # Create set of experiments
    opt = []

    neuron_multiplier = [0.25, 0.5, 1, 2, 4]
    crop_sizes = [28, 24, 20, 16, 12]
    training_data = [1]
    name = ["MLP1"]
    num_layers = [5]
    max_epochs = [100]

    idx_base = 0
    
    opt_handle = Experiments(id=idx_base, name="ColoringLSTM", dataset=opt_data[40], output_path=output_path,
                             family_id=0, family_name="Coloring_Optimal")
    opt_handle.skip_train = False
    opt_handle.dnn.name = "ColoringLSTM"
    opt_handle.dnn.n_t = 28
    opt += [copy.deepcopy(opt_handle)] 
    idx_base += 1
    
#     opt_handle = Experiments(id=idx_base, name="Coloring", dataset=opt_data[40], output_path=output_path,
#                              family_id=0, family_name="Coloring_Optimal")
#     opt_handle.skip_train = True
#     opt_handle.dnn.name = "Coloring"
#     opt_handle.dnn.n_t = 28
#     opt += [copy.deepcopy(opt_handle)] 
#     idx_base += 1

#     opt_handle = Experiments(id=idx_base, name="Coloring", dataset=opt_data[40],
#                              output_path=output_path,
#                              family_id=0, family_name="Coloring_Optimal")
#     opt_handle.skip_train = False
#     opt_handle.dnn.name = "Coloring"
#     opt_handle.dnn.n_t = 28
#     opt_handle.dnn.layers = 1
#     opt_handle.dnn.neuron_multiplier = [0.01]
#     opt += [copy.deepcopy(opt_handle)]
#     idx_base += 1

    idx_family = 1
    for idx_dataset in range(40, 50):
        for c in [5, 10, 20, 40, 80]:
            for alpha in [0.05, 0.1, 0.2]:
                for init in [1, 1e-1, 1e1]:
                    for batch in [32, 256, 2048]:
                        for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                            opt_handle = Experiments(id=idx_base, name="CrossingLearning_D" + str(idx_dataset), dataset=opt_data[idx_dataset], output_path=output_path,
                                                     family_id=idx_family, family_name="Crossing_Learning_D" + str(idx_dataset))
                            opt_handle.dnn.name = "Crossing_Learning"
                            opt_handle.hyper.complex_crossing = c
                            opt_handle.hyper.init_factor = init
                            opt_handle.hyper.max_num_epochs = 200
                            opt_handle.hyper.learning_rate = lr
                            opt_handle.hyper.alpha = alpha
                            opt_handle.hyper.batch_size = batch
                            opt += [copy.deepcopy(opt_handle)]
                            idx_base += 1

        idx_family += 1

    ''' 
    for idx in range(2):
        opt_handle = Experiments(id=idx + idx_base, name="MLP1", dataset=opt_data[0], output_path=output_path,
                                 family_id=1, family_name="A")
        opt_handle.hyper.max_num_epochs = 1

        opt += [copy.deepcopy(opt_handle)]

    idx_base = 2
    for idx in range(2):
        opt_handle = Experiments(id=idx + idx_base, name="MLP1", dataset=opt_data[0], output_path=output_path,
                                 family_id=2, family_name="B")
        opt_handle.hyper.max_num_epochs = 1

        opt += [copy.deepcopy(opt_handle)]
    '''
    return opt


def get_experiments_selected(output_path):

    NUM_TRIALS = 5

    opt_pre_cossval = get_experiments(output_path)

    with open(output_path + 'selected_models.pkl', 'rb') as f:
        cross = pickle.load(f)

    idx = 0
    opt = []
    for k in range(cross['num_families']):
        for trial in range(NUM_TRIALS):
            opt_handle = copy.deepcopy(opt_pre_cossval[cross[k]['ID']])

            opt_handle.ID = idx
            opt_handle.name = 'ID' + str(opt_handle.ID) + "_FINAL" + str(trial) + "_" + opt_handle.family_name

            idx += 1
            opt += [copy.deepcopy(opt_handle)]

    return opt
