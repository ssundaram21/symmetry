# Symmetry with deep neural networks

## Authors
* Shobhita Sundaram (shobhita@mit.edu)
* Darius Sinha
* Matthew Groth
* Tomotake Sasaki
* Xavier Boix (xboix@mit.edu)

## Object Recognition DNNs and Transformers

### Setup
Network training is implemented in Tensorflow 2.5.0. To guarante that things will successfully run, use the docker image from https://hub.docker.com/r/xboixbosch/tf2.5.0.

### Data
Training and testing datasets can be downloaded from ___.

### Running experiments (ImageNet DNNs)
To run the full experiments training/testing DNNs pretrained with ImageNet (with end-to-end finetining), run `run_full_experiment_set_1.py --idx={network-id}`. 

To run the experiments without finetuning run `run_full_experiment_set_1_no_finetuning.py --idx={network-id}`

Note that `network-id` specifies the identifier of a network-hyperparameter pairing as defined in the script. Make sure to update data, result, and model directories.

To run a demo with six object recognition DNNs pretrained on ImageNet, run this Google Colab notebook: https://colab.research.google.com/drive/1KVWLFfWGodMnS5VZrJXFplOkQeBW6Cwq#scrollTo=5Ey_wN2gaMpw

### Running experiments (CLIP Transformer)


## Dilated Convolutional Neural Network and LSTM

### Setup
Network training is implemented in Tensorflow 1.14. To guarante that things will successfully run, use the docker image from https://hub.docker.com/r/xboixbosch/tf1.14.

All networks are associated with Experiment objects. Each Experiment object has an unique identifier and defines the network, hyperparameters, and training dataset for that experiment. The Experiment objects for each network are defined in `experiments/{network name}.py`. 

The entry point for generating data and training/testing networks from scratch is `main.py`, run as detailed below.

### Data
All synthetic datasets can be created with `script-name = generate_dataset`
```
python main.py \
--experiment_index=${dataset_id} \
--code_path={your-code-path} \
--output_path={your-output-path}
--run=generate_dataset \
--network={network-name}
```
Note that the network is specified as the network that the dataset will be used for (one of `LSTM3`, `dilation`, or `multi_lstm_init`).

### Running experiments for Dilated and LSTM
The Dilated and LSTM networks can be trained as follows:
```
python main.py \
--experiment_index={experiment_id} \
--code_path={your-code-path} \
--output_path={your-output-path}
--run=train \
--network={network-name}
```
Where `experiment_id` specifies the Experiment object (defining the hyperparameters, training dataset, etc).

To test networks on test datasets, run:
```
python main.py \
--experiment_index={experiment_id} \
--code_path={your-code-path} \
--output_path={your-output-path}
--run=evaluate_generalization \
--network={network-name}
```


