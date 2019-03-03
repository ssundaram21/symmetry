import os.path
import shutil
import sys
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from nets import nets

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

    # Get data from dataset dataset
    image, y_ = iterator.get_next()
    test_in = tf.placeholder(tf.int32, [opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size])

    image = tf.cast(image, tf.int32)
    image_flat = tf.reshape(tf.cast(image, tf.int32), [opt.hyper.batch_size,  opt.dataset.image_size**2])
    test_in_flat = tf.reshape(tf.cast(test_in, tf.int32), [opt.hyper.batch_size,  opt.dataset.image_size**2])

    corr_out = tf.matmul(test_in_flat, tf.transpose(image_flat))

    with tf.Session() as sess:

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################

        sess.run(tf.global_variables_initializer())

        # TEST SET
        corr = []
        for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size) + 1):
            test_img = sess.run(image, feed_dict={handle: test_handle,
                    test_in: np.zeros((opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size))})

            ones_iter = np.sum(np.sum(test_img, axis=1), axis=1).astype(float)
            corr_iter = np.zeros(opt.hyper.batch_size).astype(float)
            for _ in range(int(dataset.num_images_training / opt.hyper.batch_size) + 1):

                corr_tmp = sess.run([corr_out], feed_dict={handle: training_handle,
                                                    test_in: test_img})

                corr_tmp = np.squeeze(np.amax(np.squeeze(corr_tmp), axis=1)).astype(float)

                corr_iter = np.maximum(corr_iter, corr_tmp/ones_iter)


            corr.append
            print("----------------")
            sys.stdout.flush()

        print(":)")
        sys.stdout.flush()



