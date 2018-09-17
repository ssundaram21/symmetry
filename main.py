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

opt = experiments.get_experiments(output_path)[FLAGS.experiment_index]


def run_train(run_opt):
    from runs import train
    train.run(run_opt)


switcher = {
    'train': run_train
}


switcher[FLAGS.run](opt)
