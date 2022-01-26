import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import datasets
import experiments
from experiments import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--code_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--raw_natural_data_path', type=str)

FLAGS = parser.parse_args()

code_path = FLAGS.code_path
output_path = FLAGS.output_path

if FLAGS.network == "dilation":
    from experiments import dilation as experiment
    output_path = output_path + "dilation/"
elif FLAGS.network == "LSTM3":
    from experiments import LSTM3 as experiment
    output_path = output_path + "lstm3/"

def run_generate_dataset(id):
    from runs import generate_dataset
    opt_data = datasets.get_datasets(output_path)[id]

    if "nat" in opt_data.name:
        if not FLAGS.raw_natural_data_path:
            raise ValueError("Path to raw natural data not specified.")
        opt_data.nat_data_path = FLAGS.raw_natural_data_path

    run_opt = experiment.generate_experiments_dataset(opt_data)
    generate_dataset.run(run_opt)

def get_dataset_as_numpy(id):
    from runs import get_dataset_as_numpy
    opt_data = datasets.get_datasets(output_path)[id]
    run_opt = experiment.generate_experiments_dataset(opt_data)
    get_dataset_as_numpy.run(run_opt)


def run_train(id):
    from runs import train
    run_opt = experiment.get_experiments(output_path)[id]
    train.run(run_opt)


def run_evaluate_generalization(id):
    opt_data = datasets.get_datasets(output_path)
    # run_opt = experiment.get_best_of_the_family(output_path)[id]
    run_opt = experiment.get_experiments(output_path)[id]
    from runs import test_generalization
    test_generalization.run(run_opt, opt_data)


def run_extract_activations(id):
    print("Getting ID:", id)
    opt_data = datasets.get_datasets(output_path)
    # run_opt = experiment.get_best_of_the_family(output_path)[id]
    run_opt = experiment.get_best_of_the_family(output_path, id)
    from runs import extract_activations
    extract_activations.run(run_opt, opt_data)


switcher = {
    'generate_dataset': run_generate_dataset,
    'generate_dataset_mix': run_generate_dataset_mix,
    'dataset_hamming': run_dataset_hamming,
    'cross_dataset_hamming': run_cross_dataset_hamming,
    'data_check': run_data_check,
    'get_dataset_as_numpy': get_dataset_as_numpy,
    'train': run_train,
    'get_train_errors': get_train_errors,
    'crossval_select': run_crossval_select,
    'train_selected': run_train_selected,
    'evaluate_generalization': run_evaluate_generalization,
    'extract_activations': run_extract_activations,
    'evaluate_perturbation': run_evaluate_perturbation
}


switcher[FLAGS.run](FLAGS.experiment_index)
