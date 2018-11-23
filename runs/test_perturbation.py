import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

from nets import nets
from util import summary
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

NUM_TRIALS = 5
DELTAS = [0, 5e-3, 1e-2, 5e-2, 1e-1, 1]

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

    if os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_perturbation.pkl'):
        print(":)")
        #quit()

    #print(opt.hyper.complex_crossing)
    print(opt.hyper.init_factor)
    print(opt.hyper.max_num_epochs)
    print(opt.hyper.learning_rate)
    print(opt.hyper.alpha)
    print(opt.hyper.batch_size)

    #tf.logging.set_verbosity(tf.logging.INFO)

    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist
    datasets = []
    test_datasets = []
    test_iterators = []
    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        for opt_dataset in opt_datasets[40:50]:
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
    delta = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name + "_Perturbation")
    y, delta_parameters, activations = to_call(image, opt, delta, len(datasets[0].list_labels)*datasets[0].num_outputs)

    # Loss function
    with tf.name_scope('loss'):

        flat_y = tf.reshape(tensor=y, shape=[-1, opt.dataset.image_size**2, len(datasets[0].list_labels)])
        flat_y_ = tf.reshape(tensor=y_, shape=[-1, opt.dataset.image_size**2])
        flat_image = tf.reshape(tensor=tf.cast(image, tf.int64), shape=[-1, opt.dataset.image_size**2])

        cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_y_, logits=flat_y)

        im = tf.cast((flat_image), tf.float32)
        cl = tf.cast(flat_y_, tf.float32)
        cross_entropy_sum = tf.reduce_mean((1-opt.hyper.alpha)*tf.reduce_sum(((1-im)*cl)*cross, 1)/tf.reduce_sum((1-im)*cl, 1) + \
                    (opt.hyper.alpha)*tf.reduce_sum(((1-im)*(1-cl)) * cross, 1) / tf.reduce_sum((1-im)*(1-cl), 1))

    # Accuracy
    with tf.name_scope('accuracy'):
        flat_output = tf.argmax(flat_y, 2)
        correct_prediction = tf.equal(flat_output * (1 - flat_image), flat_y_ * (1 - flat_image))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        error_images = tf.reduce_min(correct_prediction, 1)
        accuracy = tf.reduce_mean(error_images)

        accuracy_loose = tf.reduce_mean(
            0.5*tf.reduce_sum(((1 - im) * cl) * correct_prediction, 1) / tf.reduce_sum((1 - im) * cl, 1) + \
            0.5*tf.reduce_sum(((1 - im) * (1 - cl)) * correct_prediction, 1) / tf.reduce_sum((1 - im) * (1 - cl), 1))


    ################################################################################################


    with tf.Session() as sess:

        flag_testable = False
        if not opt.skip_train:
            ################################################################################################
            # Set up checkpoints and data
            ################################################################################################

            # Set up directories and checkpoints
            if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
                print("MODEL NOT TRAINED! ERROR")
                return

            else:
                print("RESTORE")
                sess.run(tf.initializers.variables(delta_parameters, name='init'))
                delta_parameters_names = [delta_parameter.name for delta_parameter in delta_parameters]
                restore_var = [v for v in tf.all_variables() if
                               v.name not in delta_parameters_names]  # Keep only the variables, whose name is not in the not_restore list.

                print(restore_var)
                saver = tf.train.Saver(restore_var, max_to_keep=opt.max_to_keep_checkpoints)
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
                flag_testable = True


        test_handles = []
        for test_iterator in test_iterators:
            test_handles += [sess.run(test_iterator.string_handle())]

        ################################################################################################
        # RUN TEST
        ################################################################################################

        if flag_testable:
            print("MODEL WAS NOT TRAINED")

        def test_model(data_handle, data_iterator, dataset, name, acc):
            # Run one pass over a batch of the validation dataset.

            acc[name] = {}
            acc[name]['strict'] = np.zeros([len(DELTAS), NUM_TRIALS])
            acc[name]['loose'] = np.zeros([len(DELTAS), NUM_TRIALS])
            for idx_dd, dd in enumerate(DELTAS):
                print("delta = " + str(dd))
                for tt in range(NUM_TRIALS):
                    sess.run(data_iterator.initializer)
                    sess.run(tf.initializers.variables(delta_parameters, name='init'))
                    acc_tmp = 0.0
                    acc_tmp_loo = 0.0
                    total = 0
                    for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)+1):
                        acc_val, acc_loo, a, b, err, imm = sess.run(
                            [accuracy, accuracy_loose, flat_output, y_, error_images, image],
                            feed_dict={handle: data_handle, delta: dd})

                        acc_tmp_loo += acc_loo * len(a)
                        acc_tmp += acc_val*len(a)
                        total += len(a)

                    acc[name]['strict'][idx_dd, tt] = acc_tmp / float(total)
                    acc[name]['loose'][idx_dd, tt] = acc_tmp_loo / float(total)
                    print("Full " + str(name) + " = " + str(acc[name]['strict'][idx_dd, tt]))
                    print("Full " + str(name) + " loose = " + str(acc[name]['loose'][idx_dd, tt]))
                    sys.stdout.flush()
            return acc



        import pickle
        acc = {}

        for opt_dataset, dataset, test_handle, test_iterator in \
                zip(opt_datasets, datasets, test_handles, test_iterators):

            sess.run(test_iterator.initializer)
            acc = test_model(test_handle, test_iterator, dataset, opt_dataset.ID, acc)

        if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
            os.makedirs(opt.log_dir_base + opt.name + '/results')

        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_perturbation.pkl', 'wb') as f:
            pickle.dump(acc, f)

        print(":)")



