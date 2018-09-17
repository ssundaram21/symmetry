import glob
import tensorflow as tf
import pickle
import numpy as np
from random import randint

from datasets import dataset


class FunctionDataset(dataset.Dataset):

    def __init__(self, opt):
        super(FunctionDataset, self).__init__(opt)

        self.num_threads = 8
        self.output_buffer_size = 1024

        self.list_labels = range(0, 10)
        self.num_images_training = 50000
        self.num_images_test = 10000

        self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.create_tfrecords()

    # Helper functions:
    def __unpickle(self, file_name):
        with open(file_name, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    # Virtual functions:
    def get_data_trainval(self):
        # read the 5 batch files of cifar
        addrs = []
        labels = []

        perm = np.random.permutation(32*32).astype("uint8")
        perm_x = (perm / 32).astype("uint8")
        perm_y = (perm % 32).astype("uint8")

        file_names = glob.glob(self.opt.dataset.dataset_path + "*_batch_*")
        for l in file_names:
            d = self.__unpickle(l)
            tmp = dict(d)
            X = tmp['data'].astype("uint8").reshape(10000, 3, 32, 32)
            if self.opt.dataset.scramble_data:
                X = X[:, :, perm_x, perm_y].reshape(10000, 3, 32, 32)
            X = X.transpose(0, 2, 3, 1)
            [addrs.append(l) for l in X]
            if not self.opt.dataset.random_labels:
                [labels.append(l) for l in tmp['labels']]
            else:
                [labels.append(randint(0, 9)) for l in tmp['labels']]


            train_addrs = []
            train_labels = []
            val_addrs = []
            val_labels = []

            # Divide the data into 95% train, 5% validation
            [train_addrs.append(elem) for elem in addrs[0:int(self.opt.dataset.proportion_training_set * len(addrs))]]
            [train_labels.append(elem) for elem in labels[0:int(self.opt.dataset.proportion_training_set * len(addrs))]]

            [val_addrs.append(elem) for elem in addrs[int(self.opt.dataset.proportion_training_set * len(addrs)):]]
            [val_labels.append(elem) for elem in labels[int(self.opt.dataset.proportion_training_set * len(addrs)):]]

        return train_addrs, train_labels, val_addrs, val_labels

    def get_data_test(self):
        test_addrs = []
        test_labels = []
        file_names = glob.glob(self.opt.dataset.dataset_path + "test_batch")

        perm = np.random.permutation(32*32).astype("uint8")
        perm_x = (perm / 32).astype("uint8")
        perm_y = (perm % 32).astype("uint8")

        for l in file_names:
            d = self.__unpickle(l)
            tmp = dict(d)
            X = tmp['data'].astype("uint8").reshape(10000, 3, 32, 32)
            if self.opt.dataset.scramble_data:
                X = X[:, :, perm_x, perm_y].reshape(10000, 3, 32, 32)
            X = X.transpose(0, 2, 3, 1)

            [test_addrs.append(l) for l in X]
            if not self.opt.dataset.random_labels:
                [test_labels.append(l) for l in tmp['labels']]
            else:
                [test_labels.append(randint(0, 9)) for l in tmp['labels']]

        return test_addrs, test_labels

    def preprocess_image(self, augmentation, standarization, image):
        if augmentation:
            # Randomly crop a [height, width] section of the image.
            #distorted_image = tf.random_crop(image, [self.opt.hyper.crop_size, self.opt.hyper.crop_size, 3])

            distorted_image = image
            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)
        else:
            distorted_image = image


        if standarization:
            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_standardization(distorted_image)
            float_image.set_shape([self.opt.hyper.image_size, self.opt.hyper.image_size, 3])
        else:
            float_image = distorted_image

        return float_image
