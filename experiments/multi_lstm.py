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
        self.neuron_multiplier = np.ones([self.layers])
        self.n_t = 1
        self.train_per_step = False


    def set_num_layers(self, num_layers):
        self.layers = num_layers
        self.neuron_multiplier = np.ones([self.layers])


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 1#0.95
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

    neuron_multiplier = [0.25, 0.5, 1, 2, 4]
    crop_sizes = [28, 24, 20, 16, 12]
    training_data = [1]
    name = ["MLP1"]
    num_layers = [5]
    max_epochs = [100]

    idx_base = 0
    opt_handle = Experiments(id=idx_base, name="Coloring", dataset=opt_data[49], output_path=output_path,
                             family_id=0, family_name="Coloring_Optimal")
    opt_handle.skip_train = True
    opt_handle.dnn.name = "Crossing"
    opt_handle.dnn.n_t = 30
    #opt_handle.skip = True
    opt += [copy.deepcopy(opt_handle)]
    idx_base += 1

    #810 experiments
    idx_family = 1
    for idx_dataset in range(64, 69):
        for alpha in [0.1, 0.2, 0.4]:
            for init in [0.25, 0.5, 2]:
                for batch in [32]:
                    for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                        for iterations in [5, 10, 30]:
                            opt_handle = Experiments(id=idx_base, name="MultiLSTM_D" + str(idx_dataset),
                                            dataset=opt_data[idx_dataset], output_path=output_path,
                                            family_id=idx_family, family_name="Multi_LSTM_D" + str(idx_dataset))
                            # opt_handle.dnn.name = "MultiLSTM"
                            #Coloring is the RNN
                            opt_handle.dnn.name = "Coloring"
                            # opt_handle.dnn.name = "MultiLSTMInit"
                            if batch == 32:
                                opt_handle.skip = False
                            else:
                                opt_handle.skip = True

                            # opt_handle.dnn.n_t = iterations
                            opt_handle.dnn.n_t = 60
                            opt_handle.dnn.n_t_train = iterations
                            opt_handle.dnn.train_per_step = False
                            opt_handle.hyper.init_factor = init
                            opt_handle.hyper.max_num_epochs = 10
                            opt_handle.hyper.learning_rate = lr
                            opt_handle.hyper.alpha = alpha
                            opt_handle.hyper.batch_size = batch
                            opt += [copy.deepcopy(opt_handle)]
                            idx_base += 1

        idx_family += 1

    for idx_dataset in [67]:
        for alpha in [0.1, 0.2, 0.4]:
            for init in [0.25, 0.5, 2]:
                for batch in [32]:
                    for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                        for iterations in [3, 20]:
                            opt_handle = Experiments(id=idx_base, name="MultiLSTM_D" + str(idx_dataset),
                                            dataset=opt_data[idx_dataset], output_path=output_path,
                                            family_id=idx_family, family_name="Multi_LSTM_D" + str(idx_dataset))
                            # opt_handle.dnn.name = "MultiLSTM"
                            #Coloring is the RNN
                            opt_handle.dnn.name = "Coloring"
                            # opt_handle.dnn.name = "MultiLSTMInit"
                            if batch == 32:
                                opt_handle.skip = False
                            else:
                                opt_handle.skip = True

                            # opt_handle.dnn.n_t = iterations
                            opt_handle.dnn.n_t = 60
                            opt_handle.dnn.n_t_train = iterations
                            opt_handle.dnn.train_per_step = False
                            opt_handle.hyper.init_factor = init
                            opt_handle.hyper.max_num_epochs = 10
                            opt_handle.hyper.learning_rate = lr
                            opt_handle.hyper.alpha = alpha
                            opt_handle.hyper.batch_size = batch
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

    NUM_TRIALS = 100

    opt_pre_cossval = get_experiments(output_path)

    with open(output_path + 'selected_models.pkl', 'rb') as f:
        cross = pickle.load(f)

    idx = 0
    opt = []

    for k in range(1, cross['num_families']+1):
        if not k in cross:
            continue

        for trial in range(NUM_TRIALS):
            #print(cross[k]['ID'])
            opt_handle = opt_pre_cossval[int(cross[k]['ID'])]
            opt_handle.ID = idx
            opt_handle.name = 'ID' + str(opt_handle.ID) + "_FINAL" + str(trial) + "_" + opt_handle.family_name

            idx += 1
            opt += [copy.deepcopy(opt_handle)]

    return opt
