import argparse
import datasets
import experiments


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
FLAGS = parser.parse_args()


code_path = {
    'xavier': '/Users/xboix/src/insideness/',
    'om': '/om/user/xboix/src/insideness/'}[FLAGS.host_filesystem]

output_path = {
    'xavier': '/Users/xboix/src/insideness/log/',
    'om': '/om/user/xboix/share/insideness/'}[FLAGS.host_filesystem]


def run_generate_dataset(id):
    from runs import generate_dataset
    opt_data = datasets.get_datasets(output_path)[id]
    run_opt = experiments.generate_experiments_dataset(opt_data)
    generate_dataset.run(run_opt)


def run_train(id):
    from runs import train
    run_opt = experiments.get_experiments(output_path)[id]
    train.run(run_opt)


def run_crossval_select(id):
    #id is ignored
    from runs import crossval_select
    run_opt = experiments.get_experiments(output_path)
    crossval_select.run(run_opt, output_path)


def run_train_selected(id):
    from runs import train
    run_opt = experiments.get_experiments_selected(output_path)[id]
    train.run(run_opt)


switcher = {
    'generate_dataset': run_generate_dataset,
    'train': run_train,
    'crossval_select': run_crossval_select,
    'train_selected': run_train_selected
}


switcher[FLAGS.run](FLAGS.experiment_index)
