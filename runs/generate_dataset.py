import os.path
import shutil
import sys
import numpy as np

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
'''

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

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    y, _, _ = nets.Crossing(image, opt, dropout_rate, len(dataset.list_labels)*dataset.num_outputs)
    flat_y = tf.reshape(tensor=y, shape=[-1, opt.dataset.image_size ** 2, len(dataset.list_labels)])
    flat_y_ = tf.reshape(tensor=y_, shape=[-1, opt.dataset.image_size ** 2])
    flat_image = tf.reshape(tensor=tf.cast(image, tf.int64), shape=[-1, opt.dataset.image_size ** 2])

    flat_output = tf.argmax(flat_y, 2)
    correct_prediction = tf.equal(tf.cast(flat_output * (1 - flat_image), tf.uint8), tf.cast(flat_y_,tf.uint8))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    error_images = tf.reduce_min(correct_prediction, 1)

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

        #err_idx = 0

        # Steps for doing one epoch
        for num_iter in range(int(dataset.num_images_training / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt, net_out, err_img = sess.run([image, y_, flat_output, error_images], feed_dict={handle: training_handle,
                                                       dropout_rate: 1.0})

            idx = np.asarray(list(range(len(err_img))))
            idx_acc = idx[err_img == 1]
            missing = np.uint(np.sum(1 - err_img))
            if missing > 0:
                idx_not_acc = idx_acc[np.random.randint(idx_acc.shape[0], size=missing)]
                idx[err_img == 0] = idx_not_acc

                ''' VISUALIZE ERRORS 
                mm = (err_img == 0)
                plt.imshow(np.reshape(tmp_img[mm, :, :],[100,100]))
                plt.show()
                plt.savefig(str(err_idx),
                            format="pdf", dpi=1000)
                plt.close()

                plt.imshow(np.reshape(tmp_gt[mm, :, :],[100,100]))
                plt.show()
                plt.savefig("gt"+str(err_idx),
                            format="pdf", dpi=1000)
                plt.close()

                oo = np.reshape(net_out[mm, :], [100,100])
                oo = (oo-np.reshape(tmp_gt[mm, :, :], [100,100]))*(1-np.reshape(tmp_img[mm, :, :],[100,100]))
                plt.imshow(oo)
                plt.show()
                plt.savefig("net"+str(err_idx),
                            format="pdf", dpi=1000)
                plt.close()

                err_idx = err_idx + 1
                '''

            tmp_img = tmp_img[idx, :, :]
            tmp_gt = tmp_gt[idx, :, :]

            insideness['train_img'].append(tmp_img.astype(np.uint8))
            insideness['train_gt'].append(tmp_gt.astype(np.uint8))
        insideness['train_img'] = [tmp for tmp in np.concatenate(insideness['train_img'])[:int(dataset.num_images_training), :, :]]
        insideness['train_gt'] = [tmp for tmp in np.concatenate(insideness['train_gt'])[:int(dataset.num_images_training), :, :]]

        # VALIDATION SET
        print("VALIDATION SET")
        insideness['val_img'] = []
        insideness['val_gt'] = []
        for num_iter in range(int(dataset.num_images_val / opt.hyper.batch_size) + 1):
            tmp_img, tmp_gt, err_img = sess.run([image, y_, error_images], feed_dict={handle: validation_handle,
                                                       dropout_rate: 1.0})
            idx = np.asarray(list(range(len(err_img))))
            idx_acc = idx[err_img == 1]
            missing = np.uint(np.sum(1 - err_img))
            if missing > 0:
                idx_not_acc = idx_acc[np.random.randint(idx_acc.shape[0], size=missing)]
                idx[err_img == 0] = idx_not_acc
            tmp_img = tmp_img[idx, :, :]
            tmp_gt = tmp_gt[idx, :, :]
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
            tmp_img, tmp_gt, err_img = sess.run([image, y_, error_images], feed_dict={handle: test_handle,
                                                       dropout_rate: 1.0})
            idx = np.asarray(list(range(len(err_img))))
            idx_acc = idx[err_img == 1]
            missing = np.uint(np.sum(1 - err_img))
            if missing > 0:
                idx_not_acc = idx_acc[np.random.randint(idx_acc.shape[0], size=missing)]
                idx[err_img == 0] = idx_not_acc
            tmp_img = tmp_img[idx, :, :]
            tmp_gt = tmp_gt[idx, :, :]
            insideness['test_img'].append(tmp_img.astype(np.uint8))
            insideness['test_gt'].append(tmp_gt.astype(np.uint8))
        insideness['test_img'] = [tmp for tmp in np.concatenate(insideness['test_img'])[:int(dataset.num_images_test), :, :]]
        insideness['test_gt'] = [tmp for tmp in np.concatenate(insideness['test_gt'])[:int(dataset.num_images_test), :, :]]

        # Write Ground truth
        print("WRITTING GROUNDTRUTH")
        sys.stdout.flush()

        dataset.create_tfrecords_from_numpy(insideness)

        print("----------------")
        sys.stdout.flush()

        print(":)")
        sys.stdout.flush()



