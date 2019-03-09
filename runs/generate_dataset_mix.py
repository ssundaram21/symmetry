import os.path
import shutil
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"


import pickle

import tensorflow as tf

from nets import nets

def run(opt_final, opt, opt2):

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt_final.name)
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
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)


    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
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
        validation_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################

        sess.run(tf.global_variables_initializer())

        insideness = {}

        # TRAINING SET
        print("TRAIN SET")
        insideness['train_img'] = []
        insideness['train_gt'] = []

        diff_size = opt2.dataset.image_size - opt.dataset.image_size

        # Steps for doing one epoch
        for num_iter in range(int(dataset.num_images_training / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: validation_handle})

            rr = np.random.randint(0, diff_size, size=2)

            tmp_img = np.lib.pad(tmp_img.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))
            tmp_gt = np.lib.pad(tmp_gt.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))

            insideness['train_img'].append(tmp_img.astype(np.uint8))
            insideness['train_gt'].append(tmp_gt.astype(np.uint8))

        insideness['train_img'] = [tmp for tmp in np.concatenate(insideness['train_img'])[:int(dataset.num_images_training), :, :]]
        insideness['train_gt'] = [tmp for tmp in np.concatenate(insideness['train_gt'])[:int(dataset.num_images_training), :, :]]

        #VALIDATION SET
        print("VAL SET")
        insideness['val_img'] = []
        insideness['val_gt'] = []
        # Steps for doing one epoch
        for num_iter in range(int(dataset.num_images_val / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: training_handle})

            rr = np.random.randint(0, diff_size, size=2)
            tmp_img = np.lib.pad(tmp_img.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))
            tmp_gt = np.lib.pad(tmp_gt.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))

            insideness['val_img'].append(tmp_img.astype(np.uint8))
            insideness['val_gt'].append(tmp_gt.astype(np.uint8))

        insideness['val_img'] = [tmp for tmp in np.concatenate(insideness['val_img'])[:int(dataset.num_images_val), :, :]]
        insideness['val_gt'] = [tmp for tmp in np.concatenate(insideness['val_gt'])[:int(dataset.num_images_val), :, :]]

        # TEST SET
        print("TEST SET")
        sys.stdout.flush()
        insideness['test_img'] = []
        insideness['test_gt'] = []
        for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: test_handle})

            rr = np.random.randint(0, diff_size, size=2)
            tmp_img = np.lib.pad(tmp_img.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))
            tmp_gt = np.lib.pad(tmp_gt.astype(np.uint8), ((0, 0), (rr[0], diff_size-rr[0]), (rr[1], diff_size-rr[1])), 'constant', constant_values=(0))

            insideness['test_img'].append(tmp_img.astype(np.uint8))
            insideness['test_gt'].append(tmp_gt.astype(np.uint8))

        insideness['test_img'] = [tmp for tmp in np.concatenate(insideness['test_img'])[:int(dataset.num_images_test), :, :]]
        insideness['test_gt'] = [tmp for tmp in np.concatenate(insideness['test_gt'])[:int(dataset.num_images_test), :, :]]

    tf.reset_default_graph()

    if opt2.dataset.dataset_name == 'insideness':
        from data import insideness_data
        dataset = insideness_data.InsidenessDataset(opt2)
    else:
        print("Error: no valid dataset specified")
        sys.stdout.flush()

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)


    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
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
        validation_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################

        sess.run(tf.global_variables_initializer())

        insideness2 = {}

        # TRAINING SET
        print("TRAIN SET")
        insideness2['train_img'] = []
        insideness2['train_gt'] = []

        # Steps for doing one epoch
        for num_iter in range(int(dataset.num_images_training / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: validation_handle})

            insideness2['train_img'].append(tmp_img.astype(np.uint8))
            insideness2['train_gt'].append(tmp_gt.astype(np.uint8))

        insideness2['train_img'] = [tmp for tmp in np.concatenate(insideness2['train_img'])[:int(dataset.num_images_training), :, :]]
        insideness2['train_gt'] = [tmp for tmp in np.concatenate(insideness2['train_gt'])[:int(dataset.num_images_training), :, :]]

        #VALIDATION SET
        print("VAL SET")
        insideness2['val_img'] = []
        insideness2['val_gt'] = []
        # Steps for doing one epoch
        for num_iter in range(int(dataset.num_images_val / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: training_handle})

            insideness2['val_img'].append(tmp_img.astype(np.uint8))
            insideness2['val_gt'].append(tmp_gt.astype(np.uint8))

        insideness2['val_img'] = [tmp for tmp in np.concatenate(insideness2['val_img'])[:int(dataset.num_images_val), :, :]]
        insideness2['val_gt'] = [tmp for tmp in np.concatenate(insideness2['val_gt'])[:int(dataset.num_images_val), :, :]]

        # TEST SET
        print("TEST SET")
        sys.stdout.flush()
        insideness2['test_img'] = []
        insideness2['test_gt'] = []
        for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt = sess.run([image, y_], feed_dict={handle: test_handle})

            insideness2['test_img'].append(tmp_img.astype(np.uint8))
            insideness2['test_gt'].append(tmp_gt.astype(np.uint8))

        insideness2['test_img'] = [tmp for tmp in np.concatenate(insideness2['test_img'])[:int(dataset.num_images_test), :, :]]
        insideness2['test_gt'] = [tmp for tmp in np.concatenate(insideness2['test_gt'])[:int(dataset.num_images_test), :, :]]




        insideness_final = {}
        dataset = insideness_data.InsidenessDataset(opt_final, flag_creation=False)

        insideness_final['train_img'] = np.concatenate((insideness['train_img'], insideness2['train_img']))
        print(np.shape(insideness_final['train_img']))
        insideness_final['train_gt'] = np.concatenate((insideness['train_gt'], insideness2['train_gt']))
        idx = np.random.permutation(int(dataset.num_images_training))
        insideness_final['train_img'] = insideness_final['train_img'][idx]
        insideness_final['train_gt'] = insideness_final['train_gt'][idx]

        insideness_final['val_img'] = np.concatenate((insideness['val_img'], insideness2['val_img']))
        insideness_final['val_gt'] = np.concatenate((insideness['val_gt'], insideness2['val_gt']))
        idx = np.random.permutation(int(dataset.num_images_val))
        insideness_final['val_img'] = insideness_final['val_img'][idx]
        insideness_final['val_gt'] = insideness_final['val_gt'][idx]

        insideness_final['test_img'] = np.concatenate((insideness['test_img'], insideness2['test_img']))
        insideness_final['test_gt'] = np.concatenate((insideness['test_gt'], insideness2['test_gt']))
        idx = np.random.permutation(int(dataset.num_images_test))
        insideness_final['test_img'] = insideness_final['test_img'][idx]
        insideness_final['test_gt'] = insideness_final['test_gt'][idx]

        dataset.create_tfrecords_from_numpy(insideness_final)

        print("----------------")
        sys.stdout.flush()

        print(":)")
        sys.stdout.flush()