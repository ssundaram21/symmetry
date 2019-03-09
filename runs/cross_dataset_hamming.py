import os.path
import shutil
import sys
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from nets import nets



def get_dataset_handlers(opt, opt_datasets, set):
    # Initialize dataset and creates TF records if they do not exist
    datasets = []
    test_datasets = []
    test_iterators = []
    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        for opt_dataset in opt_datasets:
            opt.dataset = opt_dataset
            datasets += [insideness_data.InsidenessDataset(opt)]
            test_datasets += [datasets[-1].create_dataset(augmentation=False, standarization=False, set_name=set,
                                                  repeat=True)]
            test_iterators += [test_datasets[-1].make_initializable_iterator()]
    else:
        print("Error: no valid dataset specified")

    return datasets, test_datasets, test_iterators


def run(opt, opt_datasets):


    opt_datasets = opt_datasets[40:50]

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt.name)
    ################################################################################################



    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################
    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        obj_dataset = insideness_data.InsidenessDataset(opt)
    else:
        print("Error: no valid dataset specified")
        sys.stdout.flush()

    test_dataset = obj_dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)
    test_iterator = test_dataset.make_initializable_iterator()


    train_obj_datasets, train_datasets, train_iterators = get_dataset_handlers(opt, opt_datasets, 'train')


    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_datasets[0].output_types, train_datasets[0].output_shapes)

    ################################################################################################

    # Get data from dataset dataset
    image, y_ = iterator.get_next()
    test_in = tf.placeholder(tf.int32, [opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size])

    image = tf.cast(image, tf.int32)
    image_flat = tf.reshape(tf.cast(image, tf.int32), [opt.hyper.batch_size,  opt.dataset.image_size**2])
    test_in_flat = tf.reshape(tf.cast(test_in, tf.int32), [opt.hyper.batch_size,  opt.dataset.image_size**2])

    corr_out = tf.matmul(test_in_flat, tf.transpose(image_flat))

    with tf.Session() as sess:

        train_handles = []
        for train_iterator in train_iterators:
            train_handles += [sess.run(train_iterator.string_handle())]

        # Run one pass over a batch of the test dataset.
        sess.run(test_iterator.initializer)
        test_handle = sess.run(test_iterator.string_handle())

        for train_opt_dataset, train_dataset, train_handle, train_iterator in \
                zip(opt_datasets, train_obj_datasets, train_handles, train_iterators):

            sess.run(train_iterator.initializer)
            training_handle = sess.run(train_iterator.string_handle())

            sess.run(tf.global_variables_initializer())

            # TEST SET

            for num_iter in range(1):#int(dataset.num_images_test / opt.hyper.batch_size) + 1):
                test_img = sess.run(image, feed_dict={handle: test_handle,
                        test_in: np.zeros((opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size))})

                ones_iter = np.sum(np.sum(test_img, axis=1), axis=1).astype(float)
                corr_iter = np.zeros(opt.hyper.batch_size).astype(float)
                corr = []
                for _ in range(int(train_dataset.num_images_training / opt.hyper.batch_size) + 1):

                    corr_tmp = sess.run([corr_out], feed_dict={handle: training_handle,
                                                        test_in: test_img})

                    corr_tmp = np.squeeze(np.amax(np.squeeze(corr_tmp), axis=1)).astype(float)

                    corr_iter = np.maximum(corr_iter, corr_tmp/ones_iter)
                    corr.append(corr_iter)

                corr = np.asarray(corr)
                corr = np.reshape(corr, np.prod(np.shape(corr)))
                print(opt.name + " vs. training on" + train_opt_dataset.name)
                print(str(np.average(corr)) + ' ' + str(np.std(corr)))

                print("----------------")
                sys.stdout.flush()

        print(":)")
        sys.stdout.flush()



