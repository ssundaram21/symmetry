# Symmetry perception by deep networks

This is the code to reproduce the results in our paper titled ["Recurrent connections facilitate symmetry perception in deep networks"](https://www.nature.com/articles/s41598-022-25219-w) (_Scientific Reports, 2022_)

## Authors
* [Shobhita Sundaram](https://ssundaram21.github.io/) (shobhita@mit.edu)
* Darius Sinha
* Matthew Groth
* Tomotake Sasaki
* [Xavier Boix](https://www.mit.edu/~xboix/) (xboix@mit.edu)

## Data
Our datasets (synthetic and natural) are available at https://dataverse.harvard.edu/dataverse/symmetry.

## Object Recognition DNNs and Transformers

### Running experiments (ImageNet DNNs)
To run a demo with six object recognition DNNs pretrained on ImageNet, run [this Google Colab notebook](https://colab.research.google.com/drive/1KVWLFfWGodMnS5VZrJXFplOkQeBW6Cwq#scrollTo=5Ey_wN2gaMpw) .

### Running experiments (CLIP Transformer)
To get synthetic and natural data for transfer-training and testing the CLIP transformer, run `Symmetry dataset.ipynb`.

To run the experiment, run `Symmetry_Interacting_with_CLIP.ipynb`.

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
--output_path={your-output-path} \
--run=generate_dataset \
--network={network-name}
```
Note that the network is specified as the network that the dataset will be used for (one of `LSTM3`, `dilation`, or `multi_lstm_init`).

To build the natural data datasets from the raw pickle files in DataVerse, specify the additional argument `--raw_natural_data_path` with the path to the pickle files.

### Running experiments for Dilated and LSTM
The Dilated and LSTM networks can be trained as follows:
```
python main.py \
--experiment_index={experiment_id} \
--code_path={your-code-path} \
--output_path={your-output-path} \
--run=train \
--network={network-name}
```
Where `experiment_id` specifies the Experiment object (defining the hyperparameters, training dataset, etc).

To test networks on test datasets, run:
```
python main.py \
--experiment_index={experiment_id} \
--code_path={your-code-path} \
--output_path={your-output-path} \
--run=evaluate_generalization \
--network={network-name}
```


