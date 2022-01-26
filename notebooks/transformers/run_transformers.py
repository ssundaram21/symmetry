import subprocess
import sys


CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)
sys.stdout.flush()

if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = ""
else:
    torch_version_suffix = "+cu110"

import numpy as np
import torch


print("Torch version:", torch.__version__)
sys.stdout.flush()

MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

model = torch.jit.load("/om/user/shobhita/src/symmetry/transformers/symmetry/model.pt").cuda().eval()
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
sys.stdout.flush()

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])


import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import pickle
import argparse

from sklearn.linear_model import LogisticRegression

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, required=True)
FLAGS = parser.parse_args()
idx = FLAGS.idx

train_sizes = [100.0, 1000.0, 10000.0, 100000.0]
train_set_list = ['synthetic', 'natural' ]
test_set_list = ['synthetic', "natural", "natural_mirrored",
                 "NS2",
                "NS6",
                "NSd4",
                "S2",
                "S6",
                "Sd4",
                "flank1S",
                "flank2S",
                "flank3S",
                "flank1NS",
                "flank2NS",
                "flank3NS",
                "stripe2S",
                "stripe4S",
                "stripe6S",
                "stripe8S",
                "stripe10S",
                "stripe2NS",
                "stripe4NS",
                "stripe6NS",
                "stripe8NS",
                "stripe10NS"]


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in DataLoader(dataset, shuffle=True, batch_size=100):
            features = model.encode_image(images.cuda())

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


PATH = '/om/user/shobhita/src/symmetry/transformers/symmetry/symmetry/'  # Make sure this path exists in your drive

training_datasets = []
for size in train_sizes:
    for dataset in train_set_list:
        if size == 100000.0 and dataset == "natural":
            continue
        else:
            training_datasets.append("{}_training_{}".format(dataset, size))

train_name = PATH + "symm_" + training_datasets[idx] + ".pkl"
with open(train_name, "rb") as handle:
    train_set = pickle.load(handle)

acc = {}
print("training")
sys.stdout.flush()
inputs = torch.tensor(
    np.stack([preprocess(Image.fromarray(x[0].astype('uint8')).convert("RGB")) for x in train_set]))
inputs -= image_mean[:, None, None]
inputs /= image_std[:, None, None]
targets = torch.IntTensor(np.stack([x[1] for x in train_set]))
train = TensorDataset(inputs, targets)
train_features, train_labels = get_features(train)

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)
del inputs
del train
del train_set
del train_features

print("testing")
sys.stdout.flush()
for test_name in test_set_list:
    with open(PATH + 'symm_' + test_name + '_test.pkl', 'rb') as handle:
        test_set = pickle.load(handle)

    inputs = torch.tensor(
        np.stack([preprocess(Image.fromarray(x[0].astype('uint8')).convert("RGB")) for x in test_set]))
    inputs -= image_mean[:, None, None]
    inputs /= image_std[:, None, None]
    targets = torch.IntTensor(np.stack([x[1] for x in test_set]))
    test = TensorDataset(inputs, targets)
    test_features, test_labels = get_features(test)

    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(train_name)
    print(test_name)
    print(f"Accuracy = {accuracy:.3f}")
    acc[test_name] = accuracy

with open(PATH + '/transformer_accuracy_{}.pkl'.format(training_datasets[idx]), 'wb') as handle:
    pickle.dump(acc, handle)