import sys
import tensorflow as tf
import datasets
import pickle
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import experiments.LSTM3 as experiments

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

output_path = args.data_path
network_id = args.model_id

output_path = output_path + 'lstm3/'
run_opt = experiments.get_best_of_the_family(output_path, network_id)
opt_datasets = datasets.get_datasets(output_path)

SYMMETRIC_DATASETS = opt_datasets[25:30] + [opt_datasets[45]] + [opt_datasets[47]] + [opt_datasets[49]]
NONSYMMETRIC_DATASETS = opt_datasets[20:25] + [opt_datasets[46]] + [opt_datasets[48]] + [opt_datasets[50]]

full_results = []
print("STARTING")

for datasets in [SYMMETRIC_DATASETS, NONSYMMETRIC_DATASETS]:
    data_points = [[] for _ in range(64)]
    output_means = {}
    output_stds = {}
    for opt_data in datasets:
        with open(run_opt.log_dir_base + run_opt.name + '/results/activations_DATA' + opt_data.log_name + '.pkl', 'rb') as f:
            data_point = pickle.load(f)
            for channel in range(64):
                for iter in range(10):
                    for time in range(50):
                        data_points[channel].append(-data_point[iter][0][5][time][:, :, :, channel])
        print('--------------------')
        print(run_opt.name)
        print(opt_data.log_name)
        sys.stdout.flush()

    for channel in range(64):
        channel_outputs = np.concatenate(data_points[channel])
        output_means[channel] = np.mean(channel_outputs, axis=0)
        output_stds[channel] = np.std(channel_outputs, axis=0)

    full_results.append({"means": output_means, "stds": output_stds})

pickle.dump(full_results, open(output_path+"activation_full_results.pkl", "wb"))

print("done :)")
