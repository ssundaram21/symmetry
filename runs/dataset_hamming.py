import os.path
import shutil
import sys
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from nets import nets

def run(opt, output_path):

    ################################################################################################
    # Read experiment to run
    ################################################################################################
    print(opt.name)

    ################################################################################################


    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    #Output path for average results
    output_path = output_path + "dataset_hamming_results/"

    #dividing factor - refers to the factor by which to reduce size of training data.
    dividing_factors = [1, 10, 100]

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        dataset = symmetry_data.SymmetryDataset(opt)
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
    print("\n\n\nIMAGE:")
    print(image.shape)
    image_flat = tf.reshape(tf.cast(image, tf.float32), [opt.hyper.batch_size,  opt.dataset.image_size**2])
    image_flat = tf.math.l2_normalize(image_flat, axis=1)
    test_in_flat = tf.reshape(tf.cast(test_in, tf.float32), [opt.hyper.batch_size,  opt.dataset.image_size**2])
    test_in_flat = tf.math.l2_normalize(test_in_flat, axis=1)
    corr_out = tf.matmul(test_in_flat, tf.transpose(image_flat))


    with tf.Session() as sess:

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        ################################################################################################
        print("\n SESSION RUN HERE")
        sess.run(tf.global_variables_initializer())

        # TEST SET
        corr = []
        full_results = {}
        for df in dividing_factors:
            corr_iters = []
            corr_iters_max = []
            corr_iters_min = []
            corr_iters_std = []
            print("BATCH SIZE: ", opt.hyper.batch_size)
            for num_iter in range(int(dataset.num_images_test / opt.hyper.batch_size)//df + 1):
                test_img = sess.run(image, feed_dict={handle: test_handle,
                        test_in: np.zeros((opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size))})

                # print(test_img)

                sys.stdout.flush()

                # ones_iter = np.sum(np.sum(test_img, axis=1), axis=1).astype(float)

                corr_iter = np.zeros(opt.hyper.batch_size).astype(float)
                for _ in range(int(dataset.num_images_training / opt.hyper.batch_size)//df + 1):

                    corr_tmp = sess.run([corr_out], feed_dict={handle: training_handle,
                                                        test_in: test_img})

                    corr_tmp = np.squeeze(np.amax(np.squeeze(corr_tmp), axis=1)).astype(float)

                    corr_iter = np.maximum(corr_iter, corr_tmp)

                print(corr_iter)
                # also look at other statistics -- max, min, sdev
                corr_iters.append(np.mean(corr_iter))
                corr_iters_max.append(np.max(corr_iter))
                corr_iters_min.append(np.min(corr_iter))
                corr_iters_std.append(np.std(corr_iter))


                print("----------------")
                sys.stdout.flush()

            #Mean of means
            total_mean = np.mean(corr_iters)
            total_max = np.max(corr_iters_max)
            total_min = np.min(corr_iters_min)
            total_std = np.mean(corr_iters_std)
            results = {
                "corr_iters_mean": corr_iters,
                "corr_iters_max": corr_iters_max,
                "corr_iters_min": corr_iters_min,
                "corr_iters_std": corr_iters_std,
                "total_avg_mean": total_mean,
                "total_avg_max": total_max,
                "total_avg_min": total_min,
                "total_avg_std": total_std

            }
            full_results[df] = results

        pickle.dump(full_results, open(output_path+"{}_HAMMING_RESULTS.p".format(dataset.opt.ID), "wb"))

        print("Done :)")
        sys.stdout.flush()



