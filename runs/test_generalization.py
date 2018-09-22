import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

from nets import nets


def run(opt, opt_datasets):

    ################################################################################################
    # Read experiment to run
    ################################################################################################

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print(opt.name)
    ################################################################################################


    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist
    datasets = []
    test_datasets = []
    test_iterators = []
    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        for opt_dataset in opt_datasets:
            opt.dataset = opt_dataset
            datasets += [insideness_data.InsidenessDataset(opt)]
            test_datasets += [datasets[-1].create_dataset(augmentation=False, standarization=False, set_name='test',
                                                  repeat=False)]
            test_iterators += [test_datasets[-1].make_initializable_iterator()]
    else:
        print("Error: no valid dataset specified")

    # No repeatable dataset for testing


    # Handles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, test_datasets[0].output_types, test_datasets[0].output_shapes)
    ################################################################################################


    ################################################################################################
    # Declare DNN
    ################################################################################################
    # Get data from dataset dataset
    image, y_ = iterator.get_next()

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, _ = to_call(image, opt, dropout_rate, len(datasets[0].list_labels)*datasets[0].num_outputs)

    flat_y = tf.reshape(tensor=y, shape=[-1, opt.dataset.image_size**2, len(datasets[0].list_labels)])
    flat_y_ = tf.reshape(tensor=y_, shape=[-1, opt.dataset.image_size**2])

    # Accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(flat_y, 2), flat_y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    ################################################################################################


    with tf.Session() as sess:
        ################################################################################################
        # Set up checkpoints and data
        ################################################################################################

        saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

        # Automatic restore model, or force train from scratch
        flag_testable = False

        # Set up directories and checkpoints
        if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
            sess.run(tf.global_variables_initializer())
        else:
            print("RESTORE")
            saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
            flag_testable = True

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        test_handles = []
        for test_iterator in test_iterators:
            test_handles += [sess.run(test_iterator.string_handle())]
        ################################################################################################

        ################################################################################################
        # RUN TEST
        ################################################################################################

        if flag_testable:

            import pickle
            acc = {}

            for opt_dataset, dataset, test_handle, test_iterator in \
                    zip(opt_datasets, datasets, test_handles, test_iterators):

                # Run one pass over a batch of the test dataset.
                sess.run(test_iterator.initializer)
                acc_tmp = 0.0
                for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)):
                    acc_val = sess.run([accuracy], feed_dict={handle: test_handle,
                                                              dropout_rate: opt.hyper.drop_test})
                    acc_tmp += acc_val[0]

                acc[opt.dataset_name] = acc_tmp / float(int(dataset.num_images_test / opt.hyper.batch_size))
                print("Full test acc: " + str(acc[opt.dataset_name]))
                sys.stdout.flush()

            if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
                os.makedirs(opt.log_dir_base + opt.name + '/results')

            with open(opt.log_dir_base + opt.name + '/results/generalization_accuracy.pkl', 'wb') as f:
                pickle.dump(acc, f)

            print(":)")

        else:
            print("MODEL WAS NOT TRAINED")

