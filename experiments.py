import numpy as np
import sys

class Dataset(object):

    def __init__(self):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.num_images_training = 1
        self.num_images_testing = 1
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        # # #
        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        self.dataset_name = "insideness"
        self.complexity = 0
        self.image_size = 100

    # # #
    # Dataset general
    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name


class DNN(object):

    def __init__(self):
        self.name = "Alexnet"
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
        self.crop_size = 28
        self.image_size = 32
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False


class Experiments(object):

    def __init__(self, id, name, output_path):
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
        self.dataset = Dataset()
        self.dnn = DNN()
        self.hyper = Hyperparameters()


def get_experiments(output_path):

    # # #
    # Create set of experiments
    opt = []

    neuron_multiplier = [0.25, 0.5, 1, 2, 4]
    crop_sizes = [28, 24, 20, 16, 12]
    training_data = [1]
    name = ["Alexnet"]
    num_layers = [5]
    max_epochs = [100]

    idx = 0
    # Create base for TF records:
    opt += [Experiments(idx, "data", output_path)]
    opt[-1].dataset.num_images_training = 1e4
    opt[-1].dataset.num_images_testing = 1e3
    opt[-1].dataset.complexity = 0
    opt[-1].hyper.max_num_epochs = 0
    idx += 1

    for name_NN, num_layers_NN, max_epochs_NN in zip(name, num_layers, max_epochs):
        for crop_size in range(len(crop_sizes)):
            opt += [Experiments(idx, name_NN + "_augmentation_" + str(crop_size), output_path)]

            opt[-1].dataset.dataset_name = 'function'

            opt[-1].hyper.max_num_epochs = max_epochs_NN
            opt[-1].hyper.crop_size = crop_size
            opt[-1].dnn.name = name_NN
            opt[-1].dnn.set_num_layers(num_layers_NN)
            opt[-1].dnn.neuron_multiplier.fill(3)

            opt[-1].dataset.reuse_tfrecords(opt[0])
            opt[-1].hyper.max_num_epochs = int(max_epochs_NN)
            opt[-1].hyper.num_epochs_per_decay = \
                int(opt[-1].hyper.num_epochs_per_decay)

            idx += 1
    return opt

