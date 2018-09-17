import argparse
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
    'xavier': '/Users/xboix/src/insideness',
    '/om': '/om/user/xboix/share/insideness/'}[FLAGS.host_filesystem]


def train(experiment_index):
    from runs import train
    train(experiment_index)

switcher = {
    'train': train
}

opt = experiments.get_experiments(code_path, output_path)[FLAGS.experiment_index]

switcher[FLAGS.run](opt)
