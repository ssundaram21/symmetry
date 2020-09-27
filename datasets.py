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

        self.dataset_name = "symmetry"
        self.type = "NS0"
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

    # 30 datasets
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        for img_type in ["NS0", "NS2", "NS4", "NS6", "NSd4", "S0", "S2", "S4", "S6", "Sd4"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = img_type

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    return opt



datasets = get_datasets("/om/user/shobhita/symmetry/get_datasets_result/")
for dataset in datasets:
    print(
        """
        \n
        id: {},
        name: {},
        dataset_path: {},
        num_images_training: {},
        num_images_testing: {},
        type: {},
        """
        .format(
            dataset.ID,
            dataset.name,
            dataset.dataset_path,
            dataset.num_images_training,
            dataset.num_images_testing,
            dataset.type
            )
        )
