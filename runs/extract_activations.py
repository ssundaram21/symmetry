import os.path
import shutil
import sys
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

from nets import nets




def get_dataset_handlers(opt, opt_datasets, name='test'):

    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist
    datasets = []
    test_datasets = []
    test_iterators = []
    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        for opt_dataset in opt_datasets:
            opt.dataset = opt_dataset
            datasets += [symmetry_data.SymmetryDataset(opt)]
            test_datasets += [datasets[-1].create_dataset(augmentation=False, standarization=False, set_name=name,
                                                  repeat=False)]
            test_iterators += [test_datasets[-1].make_initializable_iterator()]
    else:
        print("Error: no valid dataset specified")

    return datasets, test_datasets, test_iterators


def extract_activations_dataset(opt, opt_datasets, datasets, test_datasets, test_iterators, name='test'):
    # Handles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, test_datasets[0].output_types, test_datasets[0].output_shapes)
    ################################################################################################

    print(opt.dataset.image_size)
    ################################################################################################
    # Declare DNN
    ################################################################################################
    # Get data from dataset dataset
    image, y_ = iterator.get_next()
    y_ = tf.reshape(tensor=y_, shape=[opt.hyper.batch_size])

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, activations = to_call(image, opt, dropout_rate, len(datasets[0].list_labels)*datasets[0].num_outputs)

    probs = tf.nn.softmax(y)
    preds = tf.argmax(probs, 1)
    correct_prediction = tf.equal(preds, y_)

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
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/') )
                #saver.restore(sess, opt.log_dir_base + opt.name + '/models/model-' + str(opt.hyper.max_num_epochs-1))
                flag_testable = True
        else:
            sess.run(tf.global_variables_initializer())
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
            if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
                os.makedirs(opt.log_dir_base + opt.name + '/results')

            import pickle
            for opt_dataset, dataset, test_handle, test_iterator in \
                    zip(opt_datasets, datasets, test_handles, test_iterators):

                print(opt_dataset.log_name)
                # Run one pass over a batch of the test dataset.
                sess.run(test_iterator.initializer)
                total = []
                for num_iter in range(16):

                    act = sess.run([activations, image, y_, preds, correct_prediction], feed_dict={handle: test_handle,
                                                              dropout_rate: opt.hyper.drop_test})

                    total.append(act)

                sys.stdout.flush()

                if name == 'test':
                    with open(opt.log_dir_base + opt.name + '/results/activations_DATA' + opt_dataset.log_name + '.pkl', 'wb') as f:
                        print(opt.log_dir_base + opt.name + '/results/activations_DATA' + opt_dataset.log_name + '.pkl')
                        pickle.dump(total, f)
                else:
                    with open(opt.log_dir_base + opt.name + '/results/activations_' + name + '_DATA' + opt_dataset.log_name + '.pkl', 'wb') as f:
                        print(opt.log_dir_base + opt.name + '/results/activations_DATA' + opt_dataset.log_name + '.pkl')
                        pickle.dump(total, f)

        else:
            print("ERROR: MODEL WAS NOT TRAINED")



def run(opt, opt_datasets):

    ################################################################################################
    # Read experiment to run
    ################################################################################################

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print(opt.name)
    opt.hyper.batch_size = 32
    ################################################################################################

    #TODO: write a loop that goes for groups of datasets with same image size
    # Original datasets
    for i in range(20,30):
        datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[i]])
        extract_activations_dataset(opt, [opt_datasets[i]], datasets, test_datasets, test_iterators)
        tf.reset_default_graph()
    #
    # # Flank datasets
    # for i in range(45, 51):
    #     datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[i]])
    #     extract_activations_dataset(opt, [opt_datasets[i]], datasets, test_datasets, test_iterators)
    #     tf.reset_default_graph()
    #
    # # Diff datasets
    # for i in range(59, 63):
    #     datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[i]])
    #     extract_activations_dataset(opt, [opt_datasets[i]], datasets, test_datasets, test_iterators)
    #     tf.reset_default_graph()
    #
    # Stripe datasets
    # for i in range(83, 93):
    #     datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[i]])
    #     extract_activations_dataset(opt, [opt_datasets[i]], datasets, test_datasets, test_iterators)
    #     tf.reset_default_graph()


    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[52]])
    # extract_activations_dataset(opt, [opt_datasets[52]], datasets, test_datasets, test_iterators)
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[49]])
    # extract_activations_dataset(opt, [opt_datasets[49]], datasets, test_datasets, test_iterators)
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[50]])
    # extract_activations_dataset(opt, [opt_datasets[50]], datasets, test_datasets, test_iterators)
    # tf.reset_default_graph()
    #
    #
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[45]])
    # extract_activations_dataset(opt, [opt_datasets[45]], datasets, test_datasets, test_iterators)
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[41]])
    # extract_activations_dataset(opt, [opt_datasets[41]], datasets, test_datasets, test_iterators)
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[49]], 'train')
    # extract_activations_dataset(opt, [opt_datasets[49]], datasets, test_datasets, test_iterators, 'train')
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[50]], 'train')
    # extract_activations_dataset(opt, [opt_datasets[50]], datasets, test_datasets, test_iterators, 'train')
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[53]], 'train')
    # extract_activations_dataset(opt, [opt_datasets[53]], datasets, test_datasets, test_iterators, 'train')
    # tf.reset_default_graph()
    #
    # datasets, test_datasets, test_iterators = get_dataset_handlers(opt, [opt_datasets[52]], 'train')
    # extract_activations_dataset(opt, [opt_datasets[52]], datasets, test_datasets, test_iterators, 'train')
    # tf.reset_default_graph()


    print("Done :)")