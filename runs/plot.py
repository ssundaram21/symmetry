import os.path
import shutil
import sys
import numpy as np
import pickle

def run(opts, opts_dataset, output_path):


    #PLOT TRAINING ACCURACY
    acc_plot = []
    for opt in opts:
        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'rb') as f:
            acc = pickle.load(f)

            acc['train_accuracy']
