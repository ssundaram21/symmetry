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
        self.layers = 2
        self.stride = 2
        self.c = 1
        self.n_t = 1



class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 1
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.drop_train = 1
        self.drop_test = 0
        self.momentum = 0.9
        self.init_factor = 1
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
        self.restart = False

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

    idx_base = 0
    opt_handle = Experiments(id=idx_base, name="Coloring", dataset=opt_data[0], output_path=output_path,
                             family_id=0, family_name="Coloring_Optimal")
    opt_handle.skip_train = True
    opt_handle.dnn.name = "Crossing"
    opt_handle.dnn.n_t = 30
    opt += [copy.deepcopy(opt_handle)]
    idx_base += 1

    # 54 experiments
    idx_family = 1
    for idx_dataset in range(30, 33):
        for c in [4]:
            for l in [7]:
                # remove alpha
                for alpha in [0.1, 0.2, 0.4]:
                    for batch in [32]:
                        for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                            opt_handle = Experiments(id=idx_base, name="Dilation_D" + str(idx_dataset),
                                            dataset=opt_data[idx_dataset], output_path=output_path,
                                            family_id=idx_family, family_name="Dilation_D" + str(idx_dataset))
                            opt_handle.dnn.name = "Dilation"

                            if batch == 2048:
                                opt_handle.skip = True

                            opt_handle.hyper.max_num_epochs = 25
                            opt_handle.dnn.num_layers = l
                            opt_handle.dnn.complex_dilation = c
                            opt_handle.dnn.no_dilation = False
                            opt_handle.hyper.learning_rate = lr
                            opt_handle.hyper.alpha = alpha
                            opt_handle.hyper.batch_size = batch
                            opt_handle.hyper.weight_decay = 0.0
                            opt += [copy.deepcopy(opt_handle)]
                            idx_base += 1

        idx_family += 1

    # ID 55-379
    for idx_dataset in range(30, 33):
        for c in [4]:
            for l in [7]:
                for alpha in [0.1, 0.2, 0.4]:
                    for batch in [32]:
                        for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                            for noise_std in [1, 5, 8, 10, 12, 15]:
                                opt_handle = Experiments(id=idx_base, name="Dilation_D" + str(idx_dataset),
                                                         dataset=opt_data[idx_dataset], output_path=output_path,
                                                         family_id=idx_family, family_name="Dilation_D" + str(idx_dataset))
                                opt_handle.dnn.name = "Dilation"

                                if batch == 2048:
                                    opt_handle.skip = True

                                opt_handle.hyper.max_num_epochs = 25
                                opt_handle.dnn.num_layers = l
                                opt_handle.dnn.complex_dilation = c
                                opt_handle.dnn.no_dilation = False
                                opt_handle.hyper.learning_rate = lr
                                opt_handle.hyper.alpha = alpha
                                opt_handle.hyper.batch_size = batch
                                opt_handle.hyper.weight_decay = 0.0
                                opt_handle.train_with_noise = True
                                opt_handle.hyper.noise_sdev = noise_std
                                opt += [copy.deepcopy(opt_handle)]
                                idx_base += 1
        idx_family += 1

    # ID 380-398
    for idx_dataset in range(113, 114):
        for c in [4]:
            for l in [7]:
                for alpha in [0.1, 0.2, 0.4]:
                    for batch in [32]:
                        for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                            opt_handle = Experiments(id=idx_base, name="Dilation_D" + str(idx_dataset),
                                                     dataset=opt_data[idx_dataset], output_path=output_path,
                                                     family_id=idx_family,
                                                     family_name="Dilation_D" + str(idx_dataset))
                            opt_handle.dnn.name = "Dilation"

                            if batch == 2048:
                                opt_handle.skip = True

                            opt_handle.hyper.max_num_epochs = 25
                            opt_handle.dnn.num_layers = l
                            opt_handle.dnn.complex_dilation = c
                            opt_handle.dnn.no_dilation = False
                            opt_handle.hyper.learning_rate = lr
                            opt_handle.hyper.alpha = alpha
                            opt_handle.hyper.batch_size = batch
                            opt_handle.hyper.weight_decay = 0.0
                            opt += [copy.deepcopy(opt_handle)]
                            idx_base += 1
        idx_family += 1

    # ID 399-402
    idx_family = 1
    for idx_dataset in range(112, 113):
        for c in [4]:
            for l in [7]:
                # remove alpha
                for batch in [32]:
                    for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                        opt_handle = Experiments(id=idx_base, name="Dilation_D" + str(idx_dataset),
                                                 dataset=opt_data[idx_dataset], output_path=output_path,
                                                 family_id=idx_family,
                                                 family_name="Dilation_D" + str(idx_dataset))
                        opt_handle.dnn.name = "Dilation"

                        if batch == 2048:
                            opt_handle.skip = True

                        opt_handle.hyper.max_num_epochs = 25
                        opt_handle.dnn.num_layers = l
                        opt_handle.dnn.complex_dilation = c
                        opt_handle.dnn.no_dilation = False
                        opt_handle.hyper.learning_rate = lr
                        opt_handle.hyper.batch_size = batch
                        opt_handle.hyper.weight_decay = 0.0
                        opt_handle.restart = True
                        opt += [copy.deepcopy(opt_handle)]
                        idx_base += 1

    # ID 403-421
    idx_family = 1
    for idx_dataset in range(116, 119):
        for c in [4]:
            for l in [7]:
                # remove alpha
                for batch in [32]:
                    for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                        opt_handle = Experiments(id=idx_base, name="Dilation_D" + str(idx_dataset),
                                                 dataset=opt_data[idx_dataset], output_path=output_path,
                                                 family_id=idx_family,
                                                 family_name="Dilation_D" + str(idx_dataset))
                        opt_handle.dnn.name = "Dilation"

                        if batch == 2048:
                            opt_handle.skip = True

                        opt_handle.hyper.max_num_epochs = 25
                        opt_handle.dnn.num_layers = l
                        opt_handle.dnn.complex_dilation = c
                        opt_handle.dnn.no_dilation = False
                        opt_handle.hyper.learning_rate = lr
                        opt_handle.hyper.batch_size = batch
                        opt_handle.hyper.weight_decay = 0.0
                        opt_handle.restart = True
                        opt += [copy.deepcopy(opt_handle)]
                        idx_base += 1

        idx_family += 1

    return opt


def get_best_of_the_family(output_path):

    opt_pre_cossval = get_experiments(output_path)

    with open(output_path + 'selected_models.pkl', 'rb') as f:
        cross = pickle.load(f)

    opt =[]

    for k in range(1, cross['num_families']+1):
        if not k in cross:
            continue

        print(cross[k]['ID'])
        opt_handle = opt_pre_cossval[int(cross[k]['ID'])]
        opt += [copy.deepcopy(opt_handle)]

    return opt


def get_experiments_selected(output_path):

    NUM_TRIALS = 20

    opt_pre_cossval = get_experiments(output_path)

    with open(output_path + 'selected_models.pkl', 'rb') as f:
        cross = pickle.load(f)

    idx = 0
    opt = []

    for k in range(1, cross['num_families']+1):
        if not k in cross:
            continue

        opt_handle = opt_pre_cossval[int(cross[k]['ID'])]
        if opt_handle.dataset.complexity_strict == False:
            continue

        for trial in range(NUM_TRIALS):
            #print(cross[k]['ID'])
            opt_handle = opt_pre_cossval[int(cross[k]['ID'])]
            opt_handle.ID = idx
            opt_handle.name = 'ID' + str(opt_handle.ID) + "_FINAL" + str(trial) + "_" + opt_handle.family_name

            idx += 1
            opt += [copy.deepcopy(opt_handle)]

    return opt
