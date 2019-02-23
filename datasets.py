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
        self.complexity_strict = False
        self.image_size = 32

        self.name = "base"
        self.log_dir_base = output_path

        # Add ID to name:
        self.ID = id
        self.name = 'DATA_' + name
        self.log_name = 'ID' + str(id) + '_' + self.name


def get_datasets(output_path):

    # # #
    # Create set of experiments
    opt = []
    idx = 0

    for k, num_data in enumerate([1e1, 1e2, 1e3, 1e4, 1e5]):
        for complexity in range(5):
            for complexity_strict in [False, True]:
                # Create base for TF records:
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                # Create base for TF records:
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 42
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    ''' 
    opt_handle = Dataset(idx, "vanila", output_path)
    opt_handle.num_images_training = 1000
    opt_handle.num_images_testing = 100
    opt_handle.complexity = 4
    opt_handle.complexity_strict = False

    opt += [copy.deepcopy(opt_handle)]
    idx += 1
    '''

    return opt

