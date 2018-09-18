import numpy as np
import sys
import datasets
import copy


class DNN(object):

    def __init__(self):
        self.name = "MLP1"
        self.pretrained = False
        self.version = 1
        self.layers = 4
        self.stride = 2
        self.neuron_multiplier = np.ones([self.layers])

    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False


class Experiments(object):

    def __init__(self, id, name, dataset, output_path):
        self.name = "base"
        self.log_dir_base = output_path

        # Recordings
        self.max_to_keep_checkpoints = 2

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = dataset
        self.dnn = DNN()
        self.hyper = Hyperparameters()


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

    idx = 0
    opt_handle = Experiments(idx, "MLP1", opt_data[0], output_path)
    opt_handle.hyper.max_num_epochs = 100

    opt += [copy.deepcopy(opt_handle)]
    idx += 1

    return opt

