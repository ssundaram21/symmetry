import os.path
import shutil
import sys
import numpy as np

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
'''

import tensorflow as tf

from nets import nets

def run(opt):

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt.name)
    ################################################################################################


    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        dataset = symmetry_data.SymmetryDataset(opt)
    else:
        print("Error: no valid dataset specified")
        sys.stdout.flush()

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)

    print("Done :)")
    sys.stdout.flush()



