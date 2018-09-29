import datasets
import experiments
import pickle
import os
import copy
import numpy as np


def run(run_opt):

    results_data = []
    for opt in run_opt:

        if not os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl'):
            data_point = "empty"

        else:
            with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'rb') as f:
                data_point = pickle.load(f)

        results_data.append(copy.deepcopy(data_point))


    list_id_errors = []
    for point_idx, data_point in enumerate(results_data):
        if not ("val" in data_point["results"]):
            list_id_errors.append(point_idx)

    with open('error_ids', 'w') as f:
        for item in list_id_errors:
            f.write("%s\n" % item)