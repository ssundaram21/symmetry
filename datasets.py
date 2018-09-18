import numpy as np
import sys
import copy

class Dataset(object):

    def __init__(self, id, name, output_path):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.num_images_training = 1
        self.num_images_testing = 1
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        self.dataset_name = "insideness"
        self.complexity = 0
        self.image_size = 100

        self.name = "base"
        self.log_dir_base = output_path

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "DATA_" + name


def get_datasets(output_path):

    # # #
    # Create set of experiments
    opt = []
    idx = 0

    # Create base for TF records:
    opt_handle = Dataset(idx, "data", output_path)
    opt_handle.num_images_training = 1e4
    opt_handle.num_images_testing = 1e3
    opt_handle.complexity = 0

    opt += [copy.deepcopy(opt_handle)]
    idx += 1

    return opt

