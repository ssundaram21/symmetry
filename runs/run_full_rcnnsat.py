import urllib

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os
from PIL import Image
tf.compat.v1.enable_v2_behavior()

from rcnn_sat.rcnn_sat.preprocess import preprocess_image
from rcnn_sat.rcnn_sat.bl_net import bl_net

from data.generate_symmetry_images import make_random, make_images

def preprocess_ims(ims):
    return np.asarray([preprocess_image(np.asarray(Image.fromarray(x.astype("uint8")).convert("RGB").resize((128, 128)))) for x in ims])


# Replace this path with the path to Experiment Set 1 data
DATA_DIR = "/om/user/shobhita/src/symmetry/experiment_set_1/data/"

IMG_SIZE = (80, 80)
TRAIN_SIZE = 4000
TEST_CATS = [
    "band0",
    "band2",
    "band4",
    "band6",
    "bandd4",
    "band14",
    "band16",
    "band18"
]
BATCH_SIZE = 32

# /om/user/shobhita/src/symmetry/data/symm_synthetic_training_4000_80px.pkl

train_dir = os.path.join(f"{DATA_DIR}symm_synthetic_training_{TRAIN_SIZE}_{IMG_SIZE[0]}px.pkl")
with open(train_dir, "rb") as handle:
    train_dataset_raw = pickle.load(handle)

val_dir = os.path.join(f"{DATA_DIR}symm_synthetic_validation_{TRAIN_SIZE}_{IMG_SIZE[0]}px.pkl")
with open(val_dir, "rb") as handle:
    val_dataset_raw = pickle.load(handle)

test_datasets_raw = {}
for test_cat in TEST_CATS:
    test_dir = os.path.join(f"{DATA_DIR}symm_{test_cat}_{IMG_SIZE[0]}_test.pkl")
    with open(test_dir, "rb") as handle:
        test_dataset = pickle.load(handle)
    test_datasets_raw[test_cat] = test_dataset

print("Loading training dataset")
images = np.array([sample[0] for sample in train_dataset_raw])
images = preprocess_ims(images)
labels = np.array([sample[1] for sample in train_dataset_raw])
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

print("Loading validation dataset")
images = np.array([sample[0] for sample in val_dataset_raw])
images = preprocess_ims(images)
labels = np.array([sample[1] for sample in val_dataset_raw])
val_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

print("Loading test datasets")
test_datasets = {}

for test_cat, test_raw_ds in test_datasets_raw.items():
    test_images = np.array([sample[0] for sample in test_raw_ds])
    test_images = preprocess_ims(test_images)
    test_labels = np.array([sample[1] for sample in test_raw_ds])
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
    test_datasets[test_cat] = test_ds

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
for key in test_datasets:
    test_datasets[key] = test_datasets[key].prefetch(buffer_size=AUTOTUNE)


def get_model(base_model_trainable=True):
    input_layer = tf.keras.layers.Input((128, 128, 3))
    base_model = bl_net(input_layer, classes=565, cumulative_readout=True)
    base_model.load_weights('bl_imagenet.h5')
    base_model.trainable = base_model_trainable

    dense_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-6), name="Symmetry_Dense")

    dense_readout = [None for _ in range(8)]
    for ts in range(8):
        dense_readout[ts] = dense_layer(base_model.get_layer(f'GlobalAvgPool_Time_{ts}').output)

    outputs = tf.keras.layers.Average()(dense_readout)

    model = tf.keras.Model(input_layer, outputs)

    return model

model = get_model(base_model_trainable=False)
model.summary()

base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

loss0, accuracy0 = model.evaluate(validation_dataset)

history = model.fit(train_dataset,
                    epochs=30,
                    validation_data=validation_dataset)

accs = {}
for test_ds in test_datasets:
    loss, acc = model.evaluate(test_datasets[test_ds])
    accs[test_ds] = acc
    print(f"{test_ds}: {acc}")