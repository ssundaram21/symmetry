from data import data
from data import generate_shapes
import numpy as np
import random as rnd
import sys

class InsidenessDataset(data.Dataset):

    def __init__(self, opt):
        super(InsidenessDataset, self).__init__(opt)

        self.num_threads = 8

        self.num_outputs = self.opt.dataset.image_size**2
        self.list_labels = range(0, 2)
        self.num_images_training = self.opt.dataset.num_images_training
        self.num_images_test = self.opt.dataset.num_images_testing

        self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.create_tfrecords()

    def get_parameters_complexity(self, complexity, strict):

        if strict == False:

            if complexity == 0:
                num_points = [3, 5]
                minimum_radius =[23, 25]
                maximum_radius = [25, 28]

            if complexity == 1:
                num_points = [3, 10]
                minimum_radius =[19, 25]
                maximum_radius = [25, 32]

            if complexity == 2:
                num_points = [3, 15]
                minimum_radius =[15, 25]
                maximum_radius = [25, 36]

            if complexity == 3:
                num_points = [3, 20]
                minimum_radius =[10, 25]
                maximum_radius = [25, 40]

            if complexity == 4:
                num_points = [3, 25]
                minimum_radius =[6, 25]
                maximum_radius = [25, 44]

        else:
            if complexity == 0:
                num_points = [3, 5]
                minimum_radius = [23, 25]
                maximum_radius = [26, 28]

            if complexity == 1:
                num_points = [6, 10]
                minimum_radius = [19, 22]
                maximum_radius = [29, 32]

            if complexity == 2:
                num_points = [11, 15]
                minimum_radius = [15, 18]
                maximum_radius = [33, 36]

            if complexity == 3:
                num_points = [16, 20]
                minimum_radius = [10, 14]
                maximum_radius = [37, 40]

            if complexity == 4:
                num_points = [21, 25]
                minimum_radius = [6, 9]
                maximum_radius = [41, 44]

        return num_points, maximum_radius, minimum_radius


    # Virtual functions:
    def get_data_trainval(self):

        num_points_range, maximum_radius_range, minimum_radius_range = \
            self.get_parameters_complexity(self.opt.dataset.complexity, self.opt.dataset.complexity_strict)

        # read the 5 batch files of cifar
        X = []
        X_raw = []
        labels = []
        for i in range(int(self.opt.dataset.num_images_training)):
            if not i % 10:
                print('Data: {}/{}'.format(i, int(self.opt.dataset.num_images_training)))
                sys.stdout.flush()

            num_points = rnd.randint(num_points_range[0], num_points_range[1])
            minimum_radius = rnd.randint(minimum_radius_range[0], minimum_radius_range[1])
            maximum_radius = rnd.randint(maximum_radius_range[0], maximum_radius_range[1])

            img, gt, img_raw = generate_shapes.generate_data(num_points, self.opt.dataset.image_size, self.opt.dataset.image_size,
                                          maximum_radius, minimum_radius)

            X.append(np.uint8(img))

            X_raw.append(np.uint8(img_raw))
            ''' 
            from PIL import Image;
            img = Image.fromarray(128 * X[-1]);
            img.save('testrgb.png')
            '''

            labels.append(np.uint8(gt))

        train_addrs = []
        train_addrs_raw = []
        train_labels = []
        val_addrs = []
        val_addrs_raw = []
        val_labels = []

        # Divide the data into train and validation
        [train_addrs.append(elem) for elem in X[0:int(self.opt.dataset.proportion_training_set * len(X))]]
        [train_labels.append(elem) for elem in labels[0:int(self.opt.dataset.proportion_training_set * len(X))]]
        [train_addrs_raw.append(elem) for elem in X_raw[0:int(self.opt.dataset.proportion_training_set * len(X))]]

        [val_addrs.append(elem) for elem in X[int(self.opt.dataset.proportion_training_set * len(X)):]]
        [val_labels.append(elem) for elem in labels[int(self.opt.dataset.proportion_training_set * len(X)):]]
        [val_addrs_raw.append(elem) for elem in X_raw[int(self.opt.dataset.proportion_training_set * len(X)):]]

        return train_addrs, train_labels, train_addrs_raw, val_addrs, val_labels, val_addrs_raw


    def get_data_test(self):
        num_points_range, maximum_radius_range, minimum_radius_range = \
            self.get_parameters_complexity(self.opt.dataset.complexity, self.opt.dataset.complexity_strict)

        # read the 5 batch files of cifar
        X = []
        X_raw = []
        labels = []
        for i in range(int(self.opt.dataset.num_images_testing)):
            if not i % 10:
                print('Data: {}/{}'.format(i, int(self.opt.dataset.num_images_testing)))
                sys.stdout.flush()
            num_points = rnd.randint(num_points_range[0],num_points_range[1])
            minimum_radius = rnd.randint(minimum_radius_range[0], minimum_radius_range[1])
            maximum_radius = rnd.randint(maximum_radius_range[0], maximum_radius_range[1])

            img, gt, img_raw = generate_shapes.generate_data(num_points, self.opt.dataset.image_size, self.opt.dataset.image_size,
                                          maximum_radius, minimum_radius)

            X.append(np.uint8(img))
            X_raw.append(np.uint8(img_raw))
            labels.append(np.uint8(gt))

        return X, labels, X_raw


    def preprocess_image(self, augmentation, standarization, image, label, image_raw):
        image.set_shape([self.opt.dataset.image_size, self.opt.dataset.image_size])
        label.set_shape([self.opt.dataset.image_size, self.opt.dataset.image_size])
        image_raw.set_shape([self.opt.dataset.image_size, self.opt.dataset.image_size])
        return image, label, image_raw
