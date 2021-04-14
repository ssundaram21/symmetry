import glob
import pickle
import os

path = "/om/user/shobhita/data/symmetry/multi_lstm_init/dataset_hamming_results/"
directory_list = os.listdir(path)

total_results = {
    "mean":{},
    "max": {},
    "min": {},
    "std": {}
}

for file in directory_list:
    data = pickle.load(open(os.path.join(path, file), "rb"))
    id = int(file.split("_")[0])
    total_results['mean'][id] = (data[1]['total_avg_mean'], data[10]['total_avg_mean'], data[100]['total_avg_mean'])
    total_results['max'][id] = (data[1]['total_avg_max'], data[10]['total_avg_max'], data[100]['total_avg_max'])
    total_results['min'][id] = (data[1]['total_avg_min'], data[10]['total_avg_min'], data[100]['total_avg_min'])
    total_results['std'][id] = (data[1]['total_avg_std'], data[10]['total_avg_std'], data[100]['total_avg_std'])

pickle.dump(total_results, open("AGGREGATE_RESULTS.p", "wb"))
print("DONE")