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
                opt_handle.image_size = 32
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    # ID 50
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

    # ID 51
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                # Create base for TF records:
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 80
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    # ID 52
    for k, num_data in enumerate([2e5]):
        for complexity in [6]:
            for complexity_strict in [True]:
                # Create base for TF records:
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 2e4
                opt_handle.image_size = 42
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #ID 53
    for k, num_data in enumerate([1e5]):
        for complexity in [7]:
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


    #54 - Polar size 10
    for k, num_data in enumerate([1e5]):
        for complexity in [9]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 10
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1
    
    #55 - Polar size 12
    for k, num_data in enumerate([1e5]):
        for complexity in [9]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 12
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #56 - Polar size 14
    for k, num_data in enumerate([1e5]):
        for complexity in [10]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 14
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #57 - Polar size 18
    for k, num_data in enumerate([1e5]):
        for complexity in [10]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 18
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #58 - Polar size 24
    for k, num_data in enumerate([1e5]):
        for complexity in [10]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 24
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #59 - Spiral size 10
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 10
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #60 - Spiral size 12
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 12
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #61 - Spiral size 14
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 14
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #62 - Spiral size 18
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 18
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #63 - Spiral size 24
    for k, num_data in enumerate([1e5]):
        for complexity in [5]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 24
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #64 - Villim data size 8
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 8
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #65 - Villim data size 10
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 10
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #66 - Villim data size 12
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 12
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #67 - Villim data size 14
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 14
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #68 - Villim data size 18
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 18
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #69 - Villim data size 24
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 24
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    #70 - Villim data size 42
    for k, num_data in enumerate([1e5]):
        for complexity in [8]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = num_data
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 42
                opt_handle.complexity = complexity
                opt_handle.complexity_strict = complexity_strict

                opt += [copy.deepcopy(opt_handle)]
                idx += 1

    # 71 -- Symmetry 25/S0 (trial folder)
    # Complexity 9 = S0
    for k, num_data in enumerate([1e5]):
        for complexity in [12]:
            for complexity_strict in [True]:
                #Create base for TF records
                opt_handle = Dataset(idx, "C" + str(complexity) + '_' + "D" + str(k), output_path)
                opt_handle.num_images_training = 25
                opt_handle.num_images_testing = 1e4
                opt_handle.image_size = 20
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



datasets = get_datasets("/om/user/shobhita/insideness/get_datasets_result/")
for dataset in datasets:
    print(
        """
        \n
        id: {},
        name: {},
        dataset_path: {},
        num_images_training: {},
        num_images_testing: {},
        complexity: {},
        """
        .format(
            dataset.ID,
            dataset.name,
            dataset.dataset_path,
            dataset.num_images_training,
            dataset.num_images_testing,
            dataset.complexity
            )
        )
