import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

from nets import nets
from util import summary


def run(opt):

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt.name)
    ################################################################################################


    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset_name == 'insideness':
        from data import insideness_data
        dataset = insideness_data.FunctionDataset(opt)
    else:
        print("Error: no valid dataset specified")

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=False)
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=False)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=False)

    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_initializable_iterator()
    ################################################################################################


    ################################################################################################
    # Declare DNN
    ################################################################################################

    # Get data from dataset dataset
    image, y_ = iterator.get_next()

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    y, _, _ = nets.MLP1(image, dropout_rate, opt, len(dataset.list_labels)*dataset.num_outputs)


    with tf.Session() as sess:

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################

        # Steps for doing one epoch
        batch_size = 100
        for num_iter in range(int(dataset.num_images_train / batch_size)):
            tmp_gt = sess.run([y], feed_dict={handle: training_handle,
                                                       dropout_rate: 1.0})

        for num_iter in range(int(dataset.num_images_val / batch_size)):
            tmp_gt = sess.run([y], feed_dict={handle: validation_handle,
                                                       dropout_rate: 1.0})

        for num_iter in range(int(dataset.num_images_test / batch_size)):
            tmp_gt = sess.run([y], feed_dict={handle: test_handle,
                                                       dropout_rate: 1.0})


        print("----------------")
        sys.stdout.flush()
        ################################################################################################



