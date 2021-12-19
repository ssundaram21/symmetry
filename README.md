# Symmetry with deep neural networks

## Authors
* Shobhita Sundaram (shobhita@mit.edu)
* Darius Sinha
* Matthew Groth
* Tomotake Sasaki
* Xavier Boix (xboix@mit.edu)

## Setup
Network training is implemented in Tensorflow 1.14. To guarante that things will successfully run, use the docker image from https://hub.docker.com/r/xboixbosch/tf1.14.

All networks are associated with Experiment objects. Each Experiment object has an unique identifier and defines the network, hyperparameters, and training dataset for that experiment. The Experiment objects for each network are defined in `experiments/{network name}.py`. 

The entry point for generating data and running experiments is `main.py`, run as follows:
```
python /om/user/shobhita/src/symmetry/main.py \
--experiment_index=${dataset_id} \
--code_path={your-code-path} \
--output_path={your-output-path}
--run={script-name} \
--network={network-name}
```

## Data
All synthetic datasets can be created with `script-name = generate_dataset`

Note that the network is specified as the network that the dataset will be used for (one of `LSTM3`, `dilation`, or `multi_lstm_init`).

## Experiment Set 1


## Experiment Set 2
The Dilated and LSTM networks can be trained with `script-name = train`.




