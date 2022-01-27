import sys
import datasets
import pickle
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import experiments.dilation as dilated_experiments
import experiments.LSTM3 as lstm_experiments

def run(opt):
    network_id = opt.ID
    output_path = opt.log_dir_base + opt.name

    run_opt = lstm_experiments.get_best_of_the_family(output_path, network_id) if opt.name == "LSTM3_D" else dilated_experiments.get_best_of_the_family(output_path, network_id)

    opt_datasets = datasets.get_datasets(output_path)

    SYMMETRIC_DATASETS = opt_datasets[25:30] + [opt_datasets[45]] + [opt_datasets[47]] + [opt_datasets[49]]
    NONSYMMETRIC_DATASETS = opt_datasets[20:25] + [opt_datasets[46], opt_datasets[48], opt_datasets[50]]

    full_results = []
    print("STARTING")

    n = 500
    batch_size = 32
    # time = 49
    for datasets in [SYMMETRIC_DATASETS, NONSYMMETRIC_DATASETS]:
        sys.stdout.flush()
        for opt_data in datasets:
            print('--------------------')
            print("Starting ", opt_data.log_name)
            activations = []
            with open(run_opt.log_dir_base + run_opt.name + '/results/activations_DATA' + opt_data.log_name + '.pkl', 'rb') as f:
                data_point = pickle.load(f)
                count = 0
                for i in range((n // batch_size) + 1):
                    for j in range(batch_size):
                        if count == n:
                            break
                        if net == "lstm3":
                            new_dp = -data_point[i][0][-1][0][j]
                        elif net == "dilation":
                            new_dp = data_point[i][0][-2][j]
                        else:
                            raise ValueError("Invalid network specified")
                        activations.append(new_dp)
                        count += 1

            print(run_opt.name)
            print("Finished ", opt_data.log_name)
            sys.stdout.flush()

            activations = np.array(activations)
            pickle.dump(activations, open(output_path + "activation_rsa_fc_{}.pkl".format(opt_data.log_name), "wb"))

print("done :)")