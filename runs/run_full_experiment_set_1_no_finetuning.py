import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image_dataset_from_directory
import sys
import argparse
print(tf.config.list_physical_devices('GPU'))
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=int, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

DATA_DIR = args.data_path
RESULT_DIR = args.result_path
MODEL_DIR = args.model_path

BATCH_SIZE = 32
IMG_SIZE = (80, 80)
IMG_SHAPE = IMG_SIZE + (3,)
TRAIN_SIZE = 4000
MODELS = {
    "densenet": tf.keras.applications.densenet.DenseNet121,
    "nasnet": tf.keras.applications.nasnet.NASNetLarge,
    "xception": tf.keras.applications.xception.Xception,
    "inception": tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    "inception3": tf.keras.applications.inception_v3.InceptionV3,
    "resnet": tf.keras.applications.resnet.ResNet101,
    "resnet50": tf.keras.applications.resnet50.ResNet50
}
TEST_CATS = ["band0", "band2", "band4", "band6", "bandd4", "band14", "band16", "band18", "stripe2", "stripe4", "stripe6", "stripe8", "stripe10"]

## SET UP HYPERPARAMS
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, required=True)
FLAGS = parser.parse_args()
idx = FLAGS.idx

params = {}
model_id = 1
for model in ["densenet", "nasnet", "xception", "inception", "inception3", "resnet", "resnet50"]:
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
        for optimizer in ["adam"]:
            params[model_id] = {
                "batch_size": 32,
                "lr": lr,
                "n_epochs": 25,
                "model": model
            }
            model_id += 1

if idx == 0:
    model_params = {}
    batch_size = 32
    lr = 0.001
    n_epochs = 2
else:
    model_params = params[idx]
    batch_size = model_params["batch_size"]
    lr = model_params["lr"]
    n_epochs = model_params["n_epochs"]

print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Num epochs: {n_epochs}")
sys.stdout.flush()

## LOAD TRAINING/TESTING DATA
train_dir = os.path.join(f"{DATA_DIR}symm_synthetic_training_{TRAIN_SIZE}_{IMG_SIZE[0]}px.pkl")
with open(train_dir, "rb") as handle:
    train_dataset_raw = pickle.load(handle)

test_datasets_raw = {}
for test_cat in TEST_CATS:
    test_dir = os.path.join(f"{DATA_DIR}symm_{test_cat}_{IMG_SIZE[0]}_test.pkl")
    with open(test_dir, "rb") as handle:
        test_dataset = pickle.load(handle)
    test_datasets_raw[test_cat] = test_dataset

print("Loading training dataset")
images = np.array([sample[0] for sample in train_dataset_raw])
labels = np.array([sample[1] for sample in train_dataset_raw])
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

print("Loading test datasets")
test_datasets = {}
for test_raw_cat, test_raw in test_datasets_raw.items():
    test_images = np.array([sample[0] for sample in test_raw])
    test_labels = np.array([sample[1] for sample in test_raw])
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
    test_datasets[test_raw_cat] = test_dataset

## CREATE VALIDATION SET
train_batches = tf.data.experimental.cardinality(train_dataset)
val_dataset = train_dataset.take(train_batches // 10)
train_dataset = train_dataset.skip(train_batches // 10)

print(f"Number of train batches: {tf.data.experimental.cardinality(train_dataset)}")
print(f"Number of validation batches: {tf.data.experimental.cardinality(val_dataset)}")
print(f"Number of test batches: {tf.data.experimental.cardinality(test_datasets[TEST_CATS[0]])}")
sys.stdout.flush()

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
for test_cat in test_datasets:
    test_datasets[test_cat] = test_datasets[test_cat].prefetch(buffer_size=AUTOTUNE)

## SET UP MODEL
if idx == 0:
    base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
else:
    base_model = MODELS[model_params["model"]](
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet"
    )
print(f"Model: {base_model.name}")
sys.stdout.flush()
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


## FIT MODEL
history = model.fit(train_dataset,
                    epochs=n_epochs,
                    validation_data=validation_dataset)
history.history["model"] = base_model.name

model.save(MODEL_DIR + f"{base_model.name}_{idx}_model")

## TEST MODEL
test_accs = {}
for test_cat in test_datasets:
    loss, accuracy = model.evaluate(test_datasets[test_cat])
    test_accs[test_cat] = accuracy
test_accs["model"] = base_model.name

## SAVE RESULTS
with open(RESULT_DIR + f"model_{idx}_training.pkl", "wb") as handle:
    pickle.dump(history.history, handle)

with open(RESULT_DIR + f"model_{idx}_testing.pkl", "wb") as handle:
    pickle.dump(test_accs, handle)


