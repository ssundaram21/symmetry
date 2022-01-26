import numpy as np
import sys
import copy

class Dataset(object):

    def __init__(self, id, name, output_path):

        # # #
        # Dataset general
        self.dataset_path = ""
        self.nat_data_path = ""
        self.num_images_training = 1
        self.num_images_testing = 1
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        self.dataset_name = "symmetry"
        self.type = "NS0"
        self.image_size = 32

        self.name = "base"
        self.log_dir_base = output_path

        # Add ID to name:
        self.ID = id
        self.subset_no = 0
        self.name = 'DATA_' + name
        self.log_name = 'ID' + str(id) + '_' + self.name


def get_datasets(output_path):

    # # #
    # Create set of experiments
    opt = []
    idx = 0

    # ID 0-29
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        for img_type in ["NS0", "NS2", "NS4", "NS6", "NSd4", "S0", "S2", "S4", "S6", "Sd4"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 30-32
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS4", "S0", "S4"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 33-50
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        for img_type in ["flank1S", "flank1NS", "flank2S", "flank2NS", "flank3S", "flank3NS"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 51-80
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        for img_type in ["stripe2S", "stripe4S", "stripe6S", "stripe8S", "stripe10S", "stripe2NS", "stripe4NS", "stripe6NS", "stripe8NS", "stripe10NS"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 81
    # Natural image training set
    for k, num_data in enumerate([10848]):
        # Create base for TF records:
        img_type = "natTrain"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 82
    # Natural image testing set
    for k, num_data in enumerate([1200]):
        # Create base for TF records:
        img_type = "natTest"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 83
    # Natural mirrored image testing set
    for k, num_data in enumerate([1200]):
        # Create base for TF records:
        img_type = "natTestMirror"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1