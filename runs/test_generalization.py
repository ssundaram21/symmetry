import os.path
import shutil
import sys
import numpy as np
import tensorflow as tf
import pickle
from nets import nets


def get_dataset_handlers(opt, opt_datasets):
    # Initialize dataset and creates TF records if they do not exist
    datasets = []
    test_datasets = []
    test_iterators = []
    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        for opt_dataset in opt_datasets:
            opt.dataset = opt_dataset
            datasets += [symmetry_data.SymmetryDataset(opt)]
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
    y_ = tf.reshape(tensor=y_, shape=[opt.hyper.batch_size])

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, _ = to_call(image, opt, dropout_rate, len(datasets[0].list_labels)*datasets[0].num_outputs)

    with tf.name_scope('accuracy'):
        probs = tf.nn.softmax(y)
        preds = tf.argmax(probs, 1)
        correct_prediction = tf.equal(preds, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

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
            acc = {}
            acc['test_accuracy'] = {}
            for opt_dataset, dataset, test_handle, test_iterator in \
                    zip(opt_datasets, datasets, test_handles, test_iterators):

                # Run one pass over a batch of the test dataset.
                sess.run(test_iterator.initializer)
                acc_tmp = 0.0
                total = 0
                for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)+1):

                    image_val, label_val, prob_val, acc_val = sess.run([image, y_, probs, accuracy], feed_dict={handle: test_handle, dropout_rate: opt.hyper.drop_test})
                    acc_tmp += acc_val
                    total += 1

                print(total)
                acc['test_accuracy'][opt_dataset.ID] = acc_tmp / float(total)
                print("\nFor dataset {}: ".format(opt_dataset.ID, opt_dataset.type))
                print("Full test acc: " + str(acc['test_accuracy'][opt_dataset.ID]))
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
    indices = [(20, 30), (45, 51), (70, 81), (81, 84)]

    curr_datasets = opt_datasets[indices[0][0]:indices[0][1]]
    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, curr_datasets)
    acc_tmp0 = test_generalization(opt, curr_datasets, datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    curr_datasets = opt_datasets[indices[1][0]:indices[1][1]]
    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, curr_datasets)
    acc_tmp1 = test_generalization(opt, curr_datasets, datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    curr_datasets = opt_datasets[indices[2][0]:indices[2][1]]
    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, curr_datasets)
    acc_tmp2 = test_generalization(opt, curr_datasets, datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    curr_datasets = opt_datasets[indices[3][0]:indices[3][1]]
    datasets, test_datasets, test_iterators = get_dataset_handlers(opt, curr_datasets)
    acc_tmp3 = test_generalization(opt, curr_datasets, datasets, test_datasets, test_iterators)
    tf.reset_default_graph()

    results = {**acc_tmp0['test_accuracy'], **acc_tmp1['test_accuracy'], **acc_tmp2['test_accuracy'], **acc_tmp3['test_accuracy']}

    if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
        os.makedirs(opt.log_dir_base + opt.name + '/results')

    with open(opt.log_dir_base + opt.name + '/results/generalization_test.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(":)")


