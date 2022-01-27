NET = 'lstm3'
import os.path
import shutil
import sys
import tensorflow as tf
import datasets
import pickle
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import experiments.LSTM3 as experiments
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

output_path = args.data_path
network_id = args.model_id

output_path = output_path + NET + '/'
run_opt = experiments.get_best_of_the_family(output_path, network_id)
opt_datasets = datasets.get_datasets(output_path)

SYMMETRIC_DATASETS = opt_datasets[25:30] + [opt_datasets[45]] + [opt_datasets[47]] + [opt_datasets[49]]
ASYMMETRIC_DATASETS = opt_datasets[20:25] + [opt_datasets[46]] + [opt_datasets[48]] + [opt_datasets[50]]

results = {}
print("STARTING")
sys.stdout.flush()

for datasets in [SYMMETRIC_DATASETS, ASYMMETRIC_DATASETS]:
    data_points = {ts: [] for ts in [9, 19, 29, 39, 49]}
    for opt_data in datasets:
        with open(run_opt.log_dir_base + run_opt.name + '/results/activations_DATA' + opt_data.log_name + '.pkl', 'rb') as f:
            data_point = pickle.load(f)
            for channel in range(64):
                for timestep in [9,19,29,39,49]:
                    for iter in range(0, 10, 2):
                            data_points[timestep].append(-data_point[iter][0][5][timestep][:, :, :, channel])
        print('--------------------')
        print(run_opt.name)
        print(opt_data.log_name)
        sys.stdout.flush()

    for timestep in [9,19,29,39,49]:
        print("\nCLUSTERING TIMESTEP {}".format(timestep))
        dp = data_points[timestep]
        outputs = np.concatenate(dp)
        print("OUTPUT SHAPE", outputs.shape)
        sys.stdout.flush()

        o = outputs.reshape((len(outputs), 400))

        df = pd.DataFrame(o)

        print("DF SHAPE: ", df.shape)
        sys.stdout.flush()

        n = 10
        kmeans = KMeans(n_clusters=n, random_state=0).fit(df)
        results = {}
        results["centroids"] = kmeans.cluster_centers_
        results["data"] = df
        results["samples"] = [outputs[15], outputs[45], outputs[53]]
        results["labels"] = kmeans.labels_
        results["inertia"] = kmeans.inertia_

        idx = "S" if datasets == SYMMETRIC_DATASETS else "NS"
        pickle.dump(results, open(output_path+"activation_clusters_km_{}_timestep_{}_{}.pkl".format(n, timestep, idx), "wb"))

print("done :)")
