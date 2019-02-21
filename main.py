import argparse
import datasets
import experiments
from experiments import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--error_correction', type=str, default="", required=False)
FLAGS = parser.parse_args()


code_path = {
    'xavier': '/Users/xboix/src/insideness/',
    'om': '/om/user/xboix/src/insideness/',
    'om_vilim': '/om/user/vilim/src/insideness/'}[FLAGS.host_filesystem]

output_path = {
    'xavier': '/Users/xboix/src/insideness/log/',
    'om_vilim': '/om/user/xboix/share/insideness_vilim/',
    'om': '/om/user/xboix/share/insideness/'}[FLAGS.host_filesystem]


if FLAGS.network == "crossing":
    from experiments import crossing as experiment
    output_path = output_path + "crossing/"
elif FLAGS.network == "coloring":
    from experiments import coloring as experiment
    output_path = output_path + "coloring/"
elif FLAGS.network == "dilation":
    from experiments import dilation as experiment
    output_path = output_path + "dilation/"
elif FLAGS.network == "lstm":
    from experiments import lstm as experiment
    output_path = output_path + "lstm/"
elif FLAGS.network == "unet":
    from experiments import unet as experiment
    output_path = output_path + "unet/"


def run_generate_dataset(id):
    from runs import generate_dataset
    opt_data = datasets.get_datasets(output_path)[id]
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


def get_train_errors(id):
    # id is ignored
    from runs import get_train_errors
    run_opt = experiment.get_experiments(output_path)
    get_train_errors.run(run_opt)


def run_crossval_select(id):
    #id is ignored
    from runs import crossval_select
    run_opt = experiment.get_experiments(output_path)
    crossval_select.run(run_opt, output_path)


def run_train_selected(id):
    from runs import train
    run_opt = experiment.get_experiments_selected(output_path)[id]
    train.run(run_opt)


def run_evaluate_generalization(id):

    opt_data = datasets.get_datasets(output_path)
    run_opt = experiment.get_best_of_the_family(output_path)[id]

    from runs import test_generalization
    test_generalization.run(run_opt, opt_data)


def run_extract_activations(id):

    opt_data = datasets.get_datasets(output_path)
    run_opt = experiment.get_best_of_the_family(output_path)[id]

    from runs import extract_activations
    extract_activations.run(run_opt, opt_data)


def run_evaluate_perturbation(id):
    from runs import test_perturbation
    opt_data = datasets.get_datasets(output_path)
    run_opt = crossing.get_best_of_the_family(output_path)[id]
    test_perturbation.run(run_opt, opt_data)



switcher = {
    'generate_dataset': run_generate_dataset,
    'get_dataset_as_numpy': get_dataset_as_numpy,
    'train': run_train,
    'get_train_errors': get_train_errors,
    'crossval_select': run_crossval_select,
    'train_selected': run_train_selected,
    'evaluate_generalization': run_evaluate_generalization,
    'extract_activations': run_extract_activations,
    'evaluate_perturbation': run_evaluate_perturbation
}


if not (FLAGS.error_correction == ""):
    text_file = open(FLAGS.error_correction, "r")
    lines = text_file.readlines()
    id_errors = [int(l) for l in lines]
    text_file.close()

    switcher[FLAGS.run](id_errors[FLAGS.experiment_index])

else:
    switcher[FLAGS.run](FLAGS.experiment_index)
