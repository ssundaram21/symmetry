from data import data
from data import generate_symmetry_images
from data import get_natural_images
import numpy as np
import random as rnd
import sys

class SymmetryDataset(data.Dataset):

    def __init__(self, opt, flag_creation=True):
        super(SymmetryDataset, self).__init__(opt)

        self.num_threads = 8

        self.num_outputs = 1
        self.list_labels = range(0, 2)
        self.num_images_training = self.opt.dataset.num_images_training
        self.num_images_test = self.opt.dataset.num_images_testing

        self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.categories = self.opt.dataset.type

        if flag_creation:
            self.create_tfrecords()

    # Virtual functions:
    def get_data_trainval(self):
        # Complexities for each of the 4 train categories, then one for all of them mixed in.
        # read the 5 batch files of cifar
        X = []
        labels = []
        for i in range(int(self.opt.dataset.num_images_training)):
            if not i % 100:
                print('Data: {}/{}, Category: {}'.format(i, int(self.opt.dataset.num_images_training), self.opt.dataset.type))
                sys.stdout.flush()

            if self.opt.dataset.ID > 80:
                img, label = get_natural_images.get_natural_image(self.opt.dataset.type[0], i)
            else:
                if len(self.opt.dataset.type) == 1:
                    img, label = generate_symmetry_images.make_images(self.opt.dataset.type[0])
                else:
                    img, label = generate_symmetry_images.make_random(self.opt.dataset.type)
            if i % (self.opt.dataset.num_images_training / 50) == 0:
                print()
                print(label)
                print(str(img))
            X.append(np.uint8(img))
            labels.append(np.uint8(label))

        train_addrs = []
        train_labels = []
        val_addrs = []
        val_labels = []

        # Divide the data into train and validation
        [train_addrs.append(elem) for elem in X[0:int(self.opt.dataset.proportion_training_set * len(X))]]
        [train_labels.append(elem) for elem in labels[0:int(self.opt.dataset.proportion_training_set * len(X))]]

        [val_addrs.append(elem) for elem in X[int(self.opt.dataset.proportion_training_set * len(X)):]]
        [val_labels.append(elem) for elem in labels[int(self.opt.dataset.proportion_training_set * len(X)):]]

        return train_addrs, train_labels, val_addrs, val_labels


    def get_data_test(self):
        # read the 5 batch files of cifar
        X = []
        labels = []
        for i in range(int(self.opt.dataset.num_images_testing)):
            if not i % 100:
                print('Data: {}/{}, Category: {}'.format(i, int(self.opt.dataset.num_images_testing), self.opt.dataset.type))
                sys.stdout.flush()
            if self.opt.dataset.ID > 80:
                img, label = get_natural_images.get_natural_image(self.opt.dataset.type[0], i)
            else:
                if len(self.opt.dataset.type) == 1:
                    img, label = generate_symmetry_images.make_images(self.opt.dataset.type[0])
                else:
                    img, label = generate_symmetry_images.make_random(self.opt.dataset.type)

            X.append(np.uint8(img))
            labels.append(np.uint8(label))

        return X, labels


    def preprocess_image(self, augmentation, standarization, image, label):
        image.set_shape([self.opt.dataset.image_size, self.opt.dataset.image_size])
        # label.set_shape([self.opt.dataset.image_size, self.opt.dataset.image_size])
        return image, label
