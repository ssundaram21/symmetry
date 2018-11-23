import os.path
import shutil
import sys
import numpy as np

import copy
import pickle


def run(opt_all, output_path):

    selected_models = {}

    num_families = 0
    for opt in opt_all:

        # Skip execution if instructed in experiment
        if opt.skip:
            print("SKIP")
            continue

        print(opt.name)

        if not os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl'):
            print("ERROR: TRAINING NOT FINISHED")
            continue

        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'rb') as f:
            acc = pickle.load(f)

        if opt.family_ID not in selected_models:
            selected_models[opt.family_ID] = copy.deepcopy(acc)
            acc['ID'] = opt.ID
            num_families += 1
        elif selected_models[opt.family_ID]['valloose'] < acc['valloose']:
            acc['ID'] = opt.ID
            selected_models[opt.family_ID] = copy.deepcopy(acc)

    selected_models['num_families'] = num_families
    with open(output_path + 'selected_models.pkl', 'wb') as f:
        pickle.dump(selected_models, f)

    print('Num families: ' + str(num_families))
    print(":)")


