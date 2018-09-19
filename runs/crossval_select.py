import os.path
import shutil
import sys
import numpy as np

import pickle


def run(opt_all, output_path):

    selected_models = {}

    for opt in opt_all:

        # Skip execution if instructed in experiment
        if opt.skip:
            print("SKIP")
            continue

        print(opt.name)

        if not os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl'):
            print("ERROR: TRAINING NOT FINISHED")
            quit()

        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'rb') as f:
            acc = pickle.load(f)

        if opt.family_ID not in selected_models:
            selected_models[opt.family_ID] = acc
        elif selected_models[opt.family_ID]['validation_accuracy'] < acc['validation_accuracy']:
            selected_models[opt.family_ID] = acc

    with open(output_path + 'selected_models.pkl', 'wb') as f:
        pickle.dump(selected_models, f)

    print(":)")


