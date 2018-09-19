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

        if os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl'):
            print("ERROR: TRAINING NOT FINISHED")
            quit()

        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'rb') as f:
            acc = pickle.load(acc, f)

        if selected_models[opt.family_id]['val_acc'] < acc['val_acc']:
            selected_models[opt.family_id] = acc

    with open(output_path + 'selected_models.pkl', 'wb') as f:
        pickle.dump(selected_models, f)

    print(":)")


