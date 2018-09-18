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
    '/om': '/om/user/xboix/src/insideness/'}[FLAGS.host_filesystem]

output_path = {
    'xavier': '/Users/xboix/src/insideness/log/',
    '/om': '/om/user/xboix/share/insideness/'}[FLAGS.host_filesystem]


def run_train(id):
    from runs import train
    run_opt = experiments.get_experiments(output_path)[id]
    train.run(run_opt)


def generate_dataset(id):
    from runs import generate_dataset
    run_opt = datasets.get_datasets(output_path)[id]
    generate_dataset.run(run_opt)


switcher = {
    'train': run_train,
    'generate_dataset': generate_dataset
}


switcher[FLAGS.run](FLAGS.experiment_index)
