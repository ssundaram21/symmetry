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
        self.name = 'DATA' + name
        self.log_name = 'ID' + str(id) + '_' + self.name


def get_datasets(output_path):

    # # #
    # Create set of experiments
    opt = []
    idx = 0

    for k, num_data in enumerate([1e1, 1e2, 1e3, 1e4, 1e5, 1e6]):
        # Create base for TF records:
        opt_handle = Dataset(idx, "simple_" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 5e4
        opt_handle.complexity = 0

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

        opt_handle = Dataset(idx, "med-simple_" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 5e4
        opt_handle.complexity = 1

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

        opt_handle = Dataset(idx, "medium_" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 5e4
        opt_handle.complexity = 2

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

        opt_handle = Dataset(idx, "med-complex_" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 5e4
        opt_handle.complexity = 3

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

        opt_handle = Dataset(idx, "complex_" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 5e4
        opt_handle.complexity = 4

        opt += [copy.deepcopy(opt_handle)]
        idx += 1


    opt_handle = Dataset(idx, "vanila", output_path)
    opt_handle.num_images_training = 100
    opt_handle.num_images_testing = 100
    opt_handle.complexity = 4

    opt += [copy.deepcopy(opt_handle)]
    idx += 1

    return opt

