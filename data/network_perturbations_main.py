import os
import numpy as np
import tensorflow as tf
import sys
##import matplotlib.image
##from matplotlib import pyplot as plt
from generate_shapes import generate_data
from pprint import pprint

from crossing_perturbations import crossing_test

#get rid of warning msg
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Are the weights fixed? what is the perturbation amount?
# (needs more obvious implementation)
fixed_weights = True
delta = 0


#-------------------------- INITS ----------------------------------------------------------
# number of examples generated for training, testing, validation
training_size = 10
validation_size = 10
testing_size = 1000


rate_of_learning = .01
network_name = "crossing"
num_total_epochs = 1
np.set_printoptions(linewidth=400)#for debug printing purposes


n_classes = 900 #output array length
channels = 1 #no rgb
batch_size = 10
loading_print_count = []


# for dataset generation
dataset_size = training_size + validation_size + testing_size
num_points = 8
graph_width = 30
graph_height = 30
min_radius = 6
max_radius = 10


# Create the dataset
feature_list = []
for i in range(dataset_size):
    feature, filled, raw = generate_data(num_points,
                            graph_width,
                            graph_height,
                            max_radius,
                            min_radius)
    
    label_raw = filled.flatten()
    label = []
    for i in range(len(label_raw)):
        if label_raw[i] == 0:
            label.append(-1.)
        else:
            label.append(1.)

    feature_list.append((feature, np.array(label)))
    
    
#--------------------------SEPARATION AND RESHAPING-----------------------------------------
print("Separating into training, validation, and testing sets.")
training = feature_list[:training_size]
validation = feature_list[training_size: training_size + validation_size]
test = feature_list[training_size + validation_size: dataset_size]

# separate input, labels for train, test, and validation
X_train = []
Y_train = []
for data in training:
    X_train.append(data[0])
    Y_train.append(data[1])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_valid = []
Y_valid = []
for data in validation:
    X_valid.append(data[0])
    Y_valid.append(data[1])
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

X_test = []
Y_test = []
for data in test:
    X_test.append(data[0])
    Y_test.append(data[1])
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train.reshape(-1, graph_width, graph_height)
X_test = X_test.reshape(-1, graph_width, graph_height)
X_valid = X_valid.reshape(-1, graph_width, graph_height)

#-----------------------------------SET UP NETWORK VARS-----------------------------------------------------
network = crossing_test

'''
A "Dummy class". Works like a dictionary that we can potentially use to maintain separate
continuities for commonly used variables across different models. These variables will
be mapped to tensors relating to a specific network model
'''
class model_environment:
    pass

network_env = model_environment()

#--------------------------------- PLACEHOLDERS AND OUTPUT ---------------------------------------------
network_env.training = tf.placeholder(bool,
                                  shape=[],
                                  name="mode")


network_env.x = tf.placeholder(tf.float32,
                               shape=(batch_size, graph_height, graph_width),
                               name="x")

network_env.y = tf.placeholder(tf.float32,
                           shape=(batch_size, n_classes),
                           name="y")

#OUTPUT
network_env.y_pred = network(network_env.x,
                             delta,
                             fixed_weights,
                             training=network_env.training)


#------------------------------------CALCULATE ACCURACY-------------------------------------------------
target_y = tf.cast(tf.equal(network_env.y,-1),tf.float32)#set val of y to correspond w pred_y
predicted_target_y = network_env.y_pred


W = tf.contrib.layers.flatten(network_env.x)
W = tf.cast(tf.not_equal(W,1),tf.float32)


# Make it so y_pred will be equivalent to y if correct (borders don't count)
target_y = tf.multiply(target_y,W)
predicted_target_y = tf.cast(tf.greater(tf.multiply(predicted_target_y,W),.5), tf.float32)


# ACCURACY
count = tf.equal(target_y, predicted_target_y)
count = tf.cast(tf.reduce_all(count, axis=1), tf.float32)
network_env.accuracy = tf.reduce_mean(count, name="accuracy")

# LOSS
network_env.mse = tf.losses.mean_squared_error(target_y,
                                               network_env.y_pred,
                                               W)

if not fixed_weights:
    network_env.train_step = tf.train.AdamOptimizer(learning_rate=rate_of_learning).minimize(network_env.mse)

# Start session and init local & global variables --------------------------------------------------------------------------
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
current_env = network_env

# --------------------------- Training/Testing ----------------------------------------------------------------------------------------------------
'''
Judge the accuracy of a neural network
input_data: input data being tested
target_data: target data used to compare results
batch_size: size of each batch of data
environment: model_environment() class
sess: the InteractiveSession() being ran
'''
def evaluate_accuracy(input_data, target_data, batch_size, epoch, environment=current_env, session=sess):
    print("Finding accuracy...")
    accuracy = 0
    loss = 0
    samples = input_data.shape[0]
    batches = int(np.ceil(samples / batch_size))
    print("total samples:", samples)
    for i in range(batches):
        start = i * batch_size
        if start + batch_size <= samples:
            end = start + batch_size
            # use session to find environment.accuracy of each batch
            batch_loss, batch_accuracy = sess.run([environment.mse, environment.accuracy],
                                      feed_dict={environment.x: input_data[start:end],
                                                 environment.y: target_data[start:end],
                                                 environment.training: False})
        print(batch_accuracy)
        loss += batch_loss * min(end-start,batch_size)
        accuracy += batch_accuracy * min(end - start, batch_size)
    accuracy /= samples
    loss /= samples
    print("Accuracy:", accuracy)
    print("Loss:",loss)
    print()
    return accuracy


'''
Print the approximate progress made
'''
def loading_print(current_step, total_steps):
    percent = (current_step / total_steps) * 100
    if percent > len(pp):
        print(int(percent), "Percent Done ..")
        pp.append(0)
    sys.stdout.flush()


def train():
    print("Beginning Training\n")
    samples = X_train.shape[0]
    batches = int(np.ceil(samples / batch_size))
    steps = num_total_epochs
    sys.stdout.flush()
    for step in range(steps):
        print("EPOCH", step + 1)
        pp = []
        for i in range(batches):
            loading_print(i, batches)
            start = i * batch_size
            if start + batch_size < samples:
                end =  start + batch_size
                # Use session to run training step on each batch
                sess.run(current_env.train_step, feed_dict={current_env.x: X_train[start:end],
                                                            current_env.y: Y_train[start:end],
                                                            current_env.training: True})
                
        sys.stdout.flush()
        evaluate_accuracy(X_valid, Y_valid, batch_size, step, current_env)

    print("Using test data pool to evaluate accuracy\n")
    evaluate_accuracy(X_test, Y_test, batch_size, step, current_env)

def test():
    print("Using test data pool to evaluate accuracy\n")
    evaluate_accuracy(X_test, Y_test, batch_size, 1, current_env)


if fixed_weights:
    test()
else:
    train()

print("Run the command line:\n" \
      "--> tensorboard --logdir=./(dir to read from)"+
      "\nThen open http://0.0.0.0:6006/ into your web browser")
