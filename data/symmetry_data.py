from data import data
from data import generate_shapes
from data import spiral_data
from data import square
from data import small_images
import numpy as np
import random as rnd
import sys

class SymmetryDataset(data.Dataset):

    def __init__(self, opt, flag_creation=True):
        super(SymmetryDataset, self).__init__(opt)

        self.num_threads = 8

        self.num_outputs = self.opt.dataset.image_size**2
        self.list_labels = range(0, 2)
        self.num_images_training = self.opt.dataset.num_images_training
        self.num_images_test = self.opt.dataset.num_images_testing

        self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        if flag_creation:
            self.create_tfrecords()

    def get_parameters_complexity(self, complexity, strict):

        if strict == False:

            if complexity == 0:
                num_points = [3, 5]
                minimum_radius = [7, 8]
                maximum_radius = [8, 9]

            if complexity == 1:
                num_points = [3, 10]
                minimum_radius = [6, 8]
                maximum_radius = [8, 10]

            if complexity == 2:
                num_points = [3, 15]
                minimum_radius = [5, 8]
                maximum_radius = [8, 11]

            if complexity == 3:
                num_points = [3, 20]
                minimum_radius = [4, 8]
                maximum_radius = [8, 12]

            if complexity == 4:
                num_points = [3, 25]
                minimum_radius = [3, 8]
                maximum_radius = [8, 14]

            if complexity == 9:
                #For small polar datasets -- IDs 54-56
                #min/max radius get set later based on image size
                num_points = [3, 5]
                minimum_radius = 0
                maximum_radius = 0

            if complexity == 10:
                #For small polar datasets -- IDs 57-58
                #min/max radius get set later based on image size
                num_points = [3, 10]
                minimum_radius = 0
                maximum_radius = 0


        else:
            if complexity == 0:
                num_points = [3, 5]
                minimum_radius = [7, 8]
                maximum_radius = [9, 10]

            if complexity == 1:
                num_points = [6, 10]
                minimum_radius = [6, 7]
                maximum_radius = [10, 11]

            if complexity == 2:
                num_points = [11, 15]
                minimum_radius = [5, 6]
                maximum_radius = [11, 12]

            if complexity == 3:
                num_points = [16, 20]
                minimum_radius = [4, 5]
                maximum_radius = [12, 13]

            if complexity == 4:
                num_points = [21, 25]
                minimum_radius = [3, 4]
                maximum_radius = [13, 14]

            if complexity == 9:
                # For small polar datasets -- IDs 54-56
                # min/max radius get set later based on image size
                num_points = [3, 6]
                minimum_radius = 0
                maximum_radius = 0

            if complexity == 10:
                # For small polar datasets -- IDs 57-58
                # min/max radius get set later based on image size
                num_points = [3, 10]
                minimum_radius = 0
                maximum_radius = 0

        return num_points, maximum_radius, minimum_radius


    # Virtual functions:
    def get_data_trainval(self):

        if self.opt.dataset.complexity < 5 or self.opt.dataset.complexity == 9 or self.opt.dataset.complexity == 10:
            #polygon
            num_points_range, maximum_radius_range, minimum_radius_range = \
                self.get_parameters_complexity(self.opt.dataset.complexity, self.opt.dataset.complexity_strict)

        # read the 5 batch files of cifar
        X = []
        X_raw = []
        labels = []
        for i in range(int(self.opt.dataset.num_images_training)):
            if not i % 10:
                print('Data: {}/{}, Complexity: {}'.format(i, int(self.opt.dataset.num_images_training), self.opt.dataset.complexity))
                sys.stdout.flush()

            if self.opt.dataset.complexity < 5 :
                #polygon
                num_points = rnd.randint(num_points_range[0], num_points_range[1])
                minimum_radius = rnd.randint(minimum_radius_range[0], minimum_radius_range[1])
                maximum_radius = rnd.randint(maximum_radius_range[0], maximum_radius_range[1])

                img, gt, img_raw = generate_shapes.generate_data_with_check(num_points, self.opt.dataset.image_size, self.opt.dataset.image_size,
                                              maximum_radius, minimum_radius)

            elif self.opt.dataset.complexity==5:
                #spiral:
                img, gt = spiral_data.create_data_set_with_check(self.opt.dataset.image_size, self.opt.dataset.image_size)
                img_raw = img

            elif self.opt.dataset.complexity == 7:
                img = square.gen_square()
                img_raw = img
                gt = img

            elif self.opt.dataset.complexity == 8:
                img_tuple = small_images.make_small_images(
                    im_size = (self.opt.dataset.image_size, self.opt.dataset.image_size),
                    n_images=1
                )
                img = img_tuple[0].curve
                gt = img_tuple[0].inside
                img_raw = img

            elif self.opt.dataset.complexity == 9 or self.opt.dataset.complexity == 10:
                # polygon
                num_points = rnd.randint(num_points_range[0], num_points_range[1])
                minimum_radius = int(self.opt.dataset.image_size/3)
                maximum_radius = self.opt.dataset.image_size

                img, gt, img_raw = generate_shapes.generate_data_with_check(num_points, self.opt.dataset.image_size,
                                                                            self.opt.dataset.image_size,
                                                                            maximum_radius, minimum_radius)
            elif self.opt.dataset.complexity == 12:
                img = get_symmetry_data()

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
        #If necessary, set num_points_range to image_size/2
        if self.opt.dataset.complexity < 5 or self.opt.dataset.complexity == 9 or self.opt.dataset.complexity == 10:
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

            if self.opt.dataset.complexity < 5:
                num_points = rnd.randint(num_points_range[0],num_points_range[1])
                minimum_radius = rnd.randint(minimum_radius_range[0], minimum_radius_range[1])
                maximum_radius = rnd.randint(maximum_radius_range[0], maximum_radius_range[1])

                img, gt, img_raw = generate_shapes.generate_data_with_check(num_points, self.opt.dataset.image_size, self.opt.dataset.image_size,
                                              maximum_radius, minimum_radius)

            elif self.opt.dataset.complexity == 5:
                #spiral:
                img, gt = spiral_data.create_data_set(self.opt.dataset.image_size, self.opt.dataset.image_size)
                img_raw = img

            elif self.opt.dataset.complexity == 7:
                img = square.gen_square()
                img_raw = img
                gt = img

            elif self.opt.dataset.complexity == 8:
                img_tuple = small_images.make_small_images(
                    im_size = (self.opt.dataset.image_size, self.opt.dataset.image_size),
                    n_images=1
                )
                img = img_tuple[0].curve
                gt = img_tuple[0].inside
                img_raw = img

            elif self.opt.dataset.complexity == 9 or self.opt.dataset.complexity == 10:
                # polygon
                num_points = rnd.randint(num_points_range[0], num_points_range[1])
                minimum_radius = int(self.opt.dataset.image_size/3)
                maximum_radius = self.opt.dataset.image_size

                img, gt, img_raw = generate_shapes.generate_data_with_check(num_points, self.opt.dataset.image_size,
                                                                            self.opt.dataset.image_size,
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
