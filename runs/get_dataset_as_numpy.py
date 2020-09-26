import os.path
import shutil
import sys
import numpy as np

import pickle

import tensorflow as tf

from nets import nets

def run(opt):

    # Get some images and save as pkl file to pull out and check in jupyter notebook

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt.name)
    ################################################################################################


    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        dataset = insideness_data.InsidenessDataset(opt)
    else:
        print("Error: no valid dataset specified")
        sys.stdout.flush()

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)


    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()
    ################################################################################################


    ################################################################################################
    # Declare DNN
    ################################################################################################

    # Get data from dataset dataset
    image, y_ = iterator.get_next()

    with tf.Session() as sess:

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################

        sess.run(tf.global_variables_initializer())

        insideness = {}

        # TRAINING SET
        print("TRAIN SET")
        insideness['train_img'] = []
        insideness['train_gt'] = []

        # Steps for doing one epoch
        for num_iter in range(10):#int(dataset.num_images_training / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: training_handle})

            insideness['train_img'].append(tmp_img.astype(np.uint8))
            insideness['train_gt'].append(tmp_gt.astype(np.uint8))

        insideness['train_img'] = [tmp for tmp in np.concatenate(insideness['train_img'])[:int(dataset.num_images_training), :, :]]
        insideness['train_gt'] = [tmp for tmp in np.concatenate(insideness['train_gt'])[:int(dataset.num_images_training), :, :]]

        # TEST SET
        print("TEST SET")
        sys.stdout.flush()
        insideness['test_img'] = []
        insideness['test_gt'] = []
        for num_iter in range(10):#int(dataset.num_images_test / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: test_handle})

            insideness['test_img'].append(tmp_img.astype(np.uint8))
            insideness['test_gt'].append(tmp_gt.astype(np.uint8))

        insideness['test_img'] = [tmp for tmp in np.concatenate(insideness['test_img'])[:int(dataset.num_images_test), :, :]]
        insideness['test_gt'] = [tmp for tmp in np.concatenate(insideness['test_gt'])[:int(dataset.num_images_test), :, :]]

        # Write Ground truth
        print("WRITTING GROUNDTRUTH")
        sys.stdout.flush()

        print(opt.log_dir_base + opt.name + '/' + opt.name + '_dataset.pkl')
        with open(opt.log_dir_base + opt.name + '/' + opt.name + '_dataset.pkl', 'wb') as f:
            pickle.dump(insideness, f)
        print("----------------")
        sys.stdout.flush()

        print(":)")
        sys.stdout.flush()



