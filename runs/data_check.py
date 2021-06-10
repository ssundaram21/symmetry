import os.path
import shutil
import sys
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from nets import nets


def NS0(left, right):
    return (left != right).any()

def NS2(left, right):
    mid_left = left[:, -1:]
    left_side = left[:, :-1]
    mid_right = right[:, :1]
    right_side = right[:, 1:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side != right_side).any()

def NS4(left, right):
    mid_left = left[:, -2:]
    left_side = left[:, :-2]
    mid_right = right[:, :2]
    right_side = right[:, 2:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side != right_side).any()

def NS6(left, right):
    mid_left = left[:, -3:]
    left_side = left[:, :-3]
    mid_right = right[:, :3]
    right_side = right[:, 3:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side != right_side).any()

def NSd4(left, right):
    mid_left = left[:, -2:]
    left_side = left[:, :-2]
    mid_right = right[:, :2]
    right_side = right[:, 2:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side != right_side).any() and \
           (left < 129).all() and \
           (right < 129).all()

def S0(left, right):
    return (left == right).all()

def S2(left, right):
    mid_left = left[:, -1:]
    left_side = left[:, :-1]
    mid_right = right[:, :1]
    right_side = right[:, 1:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side == right_side).all()

def S4(left, right):
    mid_left = left[:, -2:]
    left_side = left[:, :-2]
    mid_right = right[:, :2]
    right_side = right[:, 2:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side == right_side).all()

def S6(left, right):
    mid_left = left[:, -3:]
    left_side = left[:, :-3]
    mid_right = right[:, :3]
    right_side = right[:, 3:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side == right_side).all()

def Sd4(left, right):
    mid_left = left[:, -2:]
    left_side = left[:, :-2]
    mid_right = right[:, :2]
    right_side = right[:, 2:]
    return (mid_left == 128).all() and \
           (mid_right == 128).all() and \
           (mid_left == mid_right).all() and \
           (left_side == right_side).all() and \
           (left < 129).all() and \
           (right < 129).all()

IMAGE_FNS = {
    "NS0": NS0,
    "NS2": NS2,
    "NS4": NS4,
    "NS6": NS6,
    "NSd4": NSd4,
    "S0": S0,
    "S2": S2,
    "S4": S4,
    "S6": S6,
    "Sd4": Sd4
}

def run(opt, output_path):
    print(opt.name)
    output_path = output_path + "data_check_results/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        dataset = symmetry_data.SymmetryDataset(opt)
    else:
        print("Error: no valid dataset specified")
        sys.stdout.flush()

    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=False)
    test_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=False)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    # Get data from dataset dataset
    image, y_ = iterator.get_next()
    test_in = tf.placeholder(tf.int32, [opt.hyper.batch_size, opt.dataset.image_size, opt.dataset.image_size])
    image = tf.cast(image, tf.int32)

    with tf.Session() as sess:

        # datasets
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        img_type = opt.dataset.type
        ################################################################################################
        print("\n SESSION RUN HERE")
        sess.run(tf.global_variables_initializer())

        # TRAIN SET
        num_wrong_train = 0
        wrong_train = []
        sample_train = []
        type_count_train = {type:0 for type in img_type}
        train_img_count = 0
        print(dataset.num_images_training)
        for iter in range(int(dataset.num_images_training / opt.hyper.batch_size)+1):
            batch = sess.run(image, feed_dict={handle: training_handle})
            for train_img in batch:
                train_img_count +=1
                mid = int(train_img.shape[0]/2)
                left = train_img[:, :mid]
                right = train_img[:, mid:]
                if len(img_type) == 1:
                    if not IMAGE_FNS[img_type[0]](left, right):
                        num_wrong_train += 1
                        wrong_train.append(train_img)
                    else:
                        type_count_train[img_type[0]] += 1
                else:
                    curr_type = None
                    for possible_type in img_type:
                        if IMAGE_FNS[possible_type](left, right):
                            type_count_train[possible_type] += 1
                            break
                    if curr_type is None:
                        num_wrong_train += 1
                        wrong_train.append(train_img)

            if (iter*opt.hyper.batch_size) % (dataset.num_images_training / 100) == 0:
                print(train_img_count)
                sample_train.append(train_img)

        # TEST SET
        num_wrong_test = 0
        wrong_test = []
        sample_test = []
        test_img_count = 0
        type_count_test = {type: 0 for type in img_type}
        for iter in range(int(dataset.num_images_test / opt.hyper.batch_size) +1):
            batch = sess.run(image, feed_dict={handle: test_handle})
            for test_img in batch:
                test_img_count += 1
                mid = int(test_img.shape[0] / 2)
                left = test_img[:, :mid]
                right = test_img[:, mid:]
                if len(img_type) == 1:
                    if not IMAGE_FNS[img_type[0]](left, right):
                        num_wrong_test += 1
                        wrong_test.append(test_img)
                    else:
                        type_count_test[img_type[0]] += 1
                else:
                    curr_type = None
                    for possible_type in img_type:
                        if IMAGE_FNS[possible_type](left, right):
                            type_count_test[possible_type] += 1
                            break
                    if curr_type is None:
                        num_wrong_test += 1
                        wrong_test.append(test_img)

            if (iter*opt.hyper.batch_size) % (dataset.num_images_test / 100) == 0:
                print(test_img_count)
                sample_test.append(test_img)

        full_results = {
            "train_img_count": train_img_count,
            "num_wrong_train": num_wrong_train,
            "wrong_train": wrong_train,
            "sample_train": sample_train,
            "type_count_train": type_count_train,
            "test_img_count": test_img_count,
            "num_wrong_test": num_wrong_test,
            "wrong_test": wrong_test,
            "sample_test": sample_test,
            "type_count_test": type_count_test
        }

        print("----------------")
        sys.stdout.flush()

        pickle.dump(full_results, open(output_path+"{}_DATA_CHECK.p".format(dataset.opt.ID), "wb"))
        print("{} train images".format(train_img_count))
        print("{} test images".format(test_img_count))
        print("{} Bad train images".format(num_wrong_train))
        print("{} Bad test images".format(num_wrong_test))
        print("Train type count: ", type_count_train)
        print("Test type count: ", type_count_test)

        print("Done :)")
        sys.stdout.flush()
