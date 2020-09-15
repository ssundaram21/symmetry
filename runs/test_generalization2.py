import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

from nets import nets


def get_dataset_handlers(opt, opt_datasets):
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

    return datasets, test_datasets, test_iterators


def test_generalization(opt, opt_datasets, datasets, test_datasets, test_iterators):
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

    flat_y = tf.reshape(tensor=y, shape=[-1, opt.dataset.image_size ** 2, len(datasets[0].list_labels)])
    flat_y_ = tf.reshape(tensor=y_, shape=[-1, opt.dataset.image_size ** 2])
    flat_image = tf.reshape(tensor=tf.cast(image, tf.int64), shape=[-1, opt.dataset.image_size ** 2])

    with tf.name_scope('accuracy'):
        flat_output = tf.argmax(flat_y, 2)
        correct_prediction = tf.equal(flat_output * (1 - flat_image), flat_y_ * (1 - flat_image))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        error_images = tf.reduce_min(correct_prediction, 1)
        accuracy = tf.reduce_mean(error_images)
        tf.summary.scalar('accuracy', accuracy)

        cl = tf.cast(flat_y_, tf.float32)
        im = tf.cast((flat_image), tf.float32)
        accuracy_loose = tf.reduce_mean(
            0.5*tf.reduce_sum(((1 - im) * cl) * correct_prediction, 1) / tf.reduce_sum((1 - im) * cl, 1) + \
            0.5*tf.reduce_sum(((1 - im) * (1 - cl)) * correct_prediction, 1) / tf.reduce_sum((1 - im) * (1 - cl), 1))


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
                #saver.restore(sess, opt.log_dir_base + opt.name + '/models/model-' + str(opt.hyper.max_num_epochs-1))

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
            acc = {}
            acc['test_accuracy'] = {}
            acc['test_accuracy_loose'] = {}
            for opt_dataset, dataset, test_handle, test_iterator in \
                    zip(opt_datasets, datasets, test_handles, test_iterators):

                # Run one pass over a batch of the test dataset.
                sess.run(test_iterator.initializer)
                acc_tmp = 0.0
                acc_tmp_loo = 0.0
                total = 0
                for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)+1):

                    acc_val, acc_loo, a = sess.run([accuracy, accuracy_loose,  flat_output], feed_dict={handle: test_handle,
                                                              dropout_rate: opt.hyper.drop_test})
                    acc_tmp += acc_val * len(a)
                    acc_tmp_loo += acc_loo * len(a)
                    total += len(a)

                print(total)
                acc['test_accuracy'][opt_dataset.ID] = acc_tmp / float(total)
                acc['test_accuracy_loose'][opt_dataset.ID] = acc_tmp_loo / float(total)
                print("Full test acc: " + str(acc['test_accuracy'][opt_dataset.ID]))
                print("Full test acc loose: " + str(acc['test_accuracy_loose'][opt_dataset.ID]))
                sys.stdout.flush()

        else:
            print("ERROR: MODEL WAS NOT TRAINED")

    return acc


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

    #TODO: write a loop that goes for groups of datasets with same image size
    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, opt_datasets[40:50])
    acc_tmp1 = test_generalization(opt, opt_datasets[40:50], datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[50]])
    acc_tmp2 = test_generalization(opt, [opt_datasets[50]], datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[51]])
    acc_tmp3 = test_generalization(opt, [opt_datasets[51]], datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[53]])
    acc_tmp4 = test_generalization(opt, [opt_datasets[53]], datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    acc={}
    acc['test_accuracy'] = {**acc_tmp1['test_accuracy'], **acc_tmp2['test_accuracy'], **acc_tmp3['test_accuracy'], **acc_tmp4['test_accuracy']}
    acc['test_accuracy_loose'] = {**acc_tmp1['test_accuracy_loose'], **acc_tmp2['test_accuracy_loose'], **acc_tmp3['test_accuracy_loose'], **acc_tmp4['test_accuracy_loose']}

    import pickle

    if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
        os.makedirs(opt.log_dir_base + opt.name + '/results')

    with open(opt.log_dir_base + opt.name + '/results/generalization_accuracy2.pkl', 'wb') as f:
        pickle.dump(acc, f)

    print(":)")


