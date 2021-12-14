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

    # ID 51-62
    for k, num_data in enumerate([1e3, 1e4, 1e5]):
        for img_type in ["diff1NS", "diff3NS", "diff5NS", "diff10NS"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 63-92
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

    # ID 93
    # 1st symmetric natural images dataset
    for k, num_data in enumerate([75]):
        for img_type in ["natS"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = num_data
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 94
    # Asymmetric natural images dataset
    for k, num_data in enumerate([57]):
        for img_type in ["natNS"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = num_data
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1


    # ID 95
    # Second symmetric natural images dataset
    for k, num_data in enumerate([176]):
        for img_type in ["natS2"]:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = num_data
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 96-106
    for k, num_data in enumerate([1e4]):
        for img_type in ["test96", "test97", "test98", "test99", "test100", "test101", "test102", "test103", "test104", "test105", "test106"]:
            # Create base for TF records:
            opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
            opt_handle.num_images_training = num_data
            opt_handle.num_images_testing = 1e4
            opt_handle.image_size = 20
            opt_handle.type = [img_type]

            opt += [copy.deepcopy(opt_handle)]
            idx += 1

    # ID 107
    for k, num_data in enumerate([1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS4Fill", "S0", "S4"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 108
    for k, num_data in enumerate([1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS18Fill", "S0", "S4"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 109
    for k, num_data in enumerate([1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS4Fill", "NS8Fill", "NS12Fill", "NS16Fill", "NS18Fill", "S0", "S0", "S0", "S0", "S0", "S0"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 110
    for k, num_data in enumerate([1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS2FillStripe", "NS4FillStripe", "NS6FillStripe", "NS8FillStripe", "NS10FillStripe", "S0", "S0", "S0", "S0", "S0"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 111
    for k, num_data in enumerate([1e5]):
        # Create base for TF records:
        img_type = "Train"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS2FillStripe1", "NS4FillStripe1", "NS6FillStripe1", "NS8FillStripe1", "NS10FillStripe1", "S0", "S0", "S0", "S0", "S0", "S0"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 112
    for k, num_data in enumerate([1e2]):
        # Create base for TF records:
        img_type = "TrainSmall"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = 1e4
        opt_handle.image_size = 20
        opt_handle.type = ["NS0", "NS4", "S0", "S4"]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 113
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

    # ID 114
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

    # ID 115
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

    # ID 116
    for k, num_data in enumerate([1e2]):
        # Create base for TF records:
        img_type = "natTrain100"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 117
    for k, num_data in enumerate([1e3]):
        # Create base for TF records:
        img_type = "natTrain1000"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 118
    for k, num_data in enumerate([1e4]):
        # Create base for TF records:
        img_type = "natTrain10000"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1



    # ID 119-128
    for k, num_data in enumerate([500]*10):
        # Create base for TF records:
        img_type = "natTestSubset"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.subset_no = k
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

        opt += [copy.deepcopy(opt_handle)]
        idx += 1

    # ID 129-138
    for k, num_data in enumerate([500]*10):
        # Create base for TF records:
        img_type = "natTestMirrorSubset"
        opt_handle = Dataset(idx, "Cat" + str(img_type) + '_' + "D" + str(k), output_path)
        opt_handle.subset_no = k
        opt_handle.num_images_training = num_data
        opt_handle.num_images_testing = num_data
        opt_handle.image_size = 20
        opt_handle.type = [img_type]

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
