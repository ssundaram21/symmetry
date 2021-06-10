import urllib

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
# import matplotlib.pyplot as plt
from PIL import Image
import sys
tf.compat.v1.enable_v2_behavior()

from rcnn_sat.rcnn_sat.preprocess import preprocess_image
from rcnn_sat.rcnn_sat.bl_net import bl_net

from data.generate_symmetry_images import make_random, make_images
from sklearn.linear_model import LogisticRegression

data_path = "/om/user/shobhita/data/symmetry/rcnn_sat/"
test_sets = ["NS0", "NS2", "NS4", "NS6", "NSd4", "S0", "S2", "S4", "S6", "Sd4"]

input_layer = tf.keras.layers.Input((128, 128, 3))
model = bl_net(input_layer, classes=565, cumulative_readout=True)

_, msg = urllib.request.urlretrieve(
    'https://osf.io/9td5p/download', 'bl_imagenet.h5')
print(msg)
sys.stdout.flush()

model.load_weights('bl_imagenet.h5')

activation_fns = []
for ts in range(8):
    activation_fns.append(
        tf.keras.backend.function(
            [model.input],
            [model.get_layer('GlobalAvgPool_Time_{}'.format(ts)).output])
    )

print("Loading data")
sys.stdout.flush()
train_data_synthetic = pickle.load(open(data_path + "rcnn_train.pkl", "rb"))
train_ims, train_labels = train_data_synthetic
train_size = 10000
indices = np.random.choice(len(train_ims), train_size, replace=False).astype('uint8')
train_ims, train_labels = np.asarray(train_ims), np.asarray(train_labels)
train_ims, train_labels = train_ims[indices], train_labels[indices]


def get_activations(ims, batch_size=32):
    processed_ims = np.asarray(
        [preprocess_image(np.asarray(Image.fromarray(x.astype("uint8")).convert("RGB").resize((128, 128)))) for x in
         ims])

    activations = []
    n_steps = len(processed_ims) // batch_size
    for i in range(n_steps):
        if i % 10 == 0:
            print("Batch {}/{}".format(i + 1, n_steps))
            sys.stdout.flush()
        batch_ims = processed_ims[i * batch_size : (i+1) * batch_size]
        batch_acts = []
        for ts in range(8):
            batch_act = activation_fns[ts](batch_ims)[0]
            batch_acts.append(batch_act)
        activations.append(np.concatenate(np.asarray(batch_acts), axis=1))

    if n_steps * batch_size < len(processed_ims):
        print("Batch {}".format(n_steps + 1))
        sys.stdout.flush()
        final_batch_ims = processed_ims[n_steps * batch_size:]
        batch_acts = []
        for ts in range(8):
            batch_act = activation_fns[ts](final_batch_ims)[0]
            batch_acts.append(batch_act)
        activations.append(np.concatenate(np.asarray(batch_acts), axis=1))

    return np.concatenate(activations)

print("Getting training activations")
sys.stdout.flush()
trainx = get_activations(train_ims)
trainy = train_labels

print("Training")
sys.stdout.flush()
model = LogisticRegression(max_iter=1000)
model_fit = model.fit(trainx, trainy)

results = {}

print("Testing")
sys.stdout.flush()
for ds in test_sets:
    test_ims, test_labels = pickle.load(open(data_path + "rcnn_{}_test.pkl".format(ds), "rb"))
    test_ims, test_labels = test_ims, test_labels
    testx = get_activations(test_ims)
    testy = test_labels
    probs = model_fit.predict_proba(testx)
    preds = np.argmax(probs, axis=1)
    acc = np.sum(preds == testy) / len(testy)
    results[ds] = {'acc': acc, 'probs': probs}
    print("Dataset {}: {}".format(ds, acc))
    sys.stdout.flush()

pickle.dump(results, open(data_path + "rcnn_synthetic_to_synthetic_size_{}.pkl".format(train_size), "wb"))







