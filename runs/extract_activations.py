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
                                                  repeat=True)]
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
    y, parameters, activations = to_call(image, opt, dropout_rate, len(datasets[0].list_labels)*datasets[0].num_outputs)



    ################################################################################################


    with tf.Session() as sess:
        ################################################################################################
        # Set up checkpoints and data
        ################################################################################################

        # Automatic restore model, or force train from scratch
        flag_testable = False
        if not opt.skip_train:

            saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

            # Set up directories and checkpoints
            if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
                sess.run(tf.global_variables_initializer())
            else:
                print("RESTORE")
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
                flag_testable = True
        else:
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
            acc['test_accuracy'] = {}
            acc['test_accuracy_loose'] = {}
            total = {}
            for opt_dataset, dataset, test_handle, test_iterator in \
                    zip(opt_datasets, datasets, test_handles, test_iterators):

                # Run one pass over a batch of the test dataset.
                sess.run(test_iterator.initializer)
                total[opt_dataset.log_name] = []
                for num_iter in range(1000):

                    act = sess.run(activations, feed_dict={handle: test_handle,
                                                              dropout_rate: opt.hyper.drop_test})

                    total[opt_dataset.log_name].append(act)


                sys.stdout.flush()

            if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
                os.makedirs(opt.log_dir_base + opt.name + '/results')

            with open(opt.log_dir_base + opt.name + '/results/activations.pkl', 'wb') as f:
                pickle.dump(total, f)

            print(":)")

        else:
            print("ERROR: MODEL WAS NOT TRAINED")

