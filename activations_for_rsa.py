NET = 'lstm3'

import sys
import datasets
import pickle
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--idx', type=int, required=True)

FLAGS = parser.parse_args()

NET = FLAGS.network
network_id = FLAGS.idx

if NET == 'dilation':
    import experiments.dilation as experiments
if NET == 'segnet':
    import experiments.segnet as experiments
elif NET == 'lstm':
    import experiments.lstm as experiments
elif NET == 'coloring':
    import experiments.coloring as experiments
elif NET == 'crossing':
    import experiments.crossing as experiments
elif NET == 'unet':
    import experiments.unet as experiments
elif NET == 'multi_lstm':
    import experiments.multi_lstm as experiments
elif NET == 'multi_lstm_init':
    import experiments.multi_lstm_init as experiments
elif NET == 'FF':
    import experiments.FF as experiments
elif NET == 'optimal_lstm':
    import experiments.optimal_lstm as experiments
elif NET == 'LSTM3':
    import experiments.LSTM3 as experiments

output_path = '/om/user/shobhita/data/symmetry/' + NET.lower() + '/'
run_opt = experiments.get_best_of_the_family(output_path, network_id)
opt_datasets = datasets.get_datasets(output_path)

SYMMETRIC_DATASETS = opt_datasets[25:30] + [opt_datasets[45], opt_datasets[47], opt_datasets[49]]
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
                    if NET == "LSTM3":
                        new_dp = -data_point[i][0][-1][0][j]
                    else:
                        new_dp = data_point[i][0][-2][j]
                    assert new_dp.shape[0] == 512
                    activations.append(new_dp)
                    count += 1

        print(run_opt.name)
        print("Finished ", opt_data.log_name)
        sys.stdout.flush()

        activations = np.array(activations)
        pickle.dump(activations, open(output_path + "activation_rsa_fc_{}.pkl".format(opt_data.log_name), "wb"))

print("done :)")
