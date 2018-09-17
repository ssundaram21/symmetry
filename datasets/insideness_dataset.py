from datasets import dataset
import generate_dataset
import numpy as np

class FunctionDataset(dataset.Dataset):

    def __init__(self, opt):
        super(FunctionDataset, self).__init__(opt)

        self.num_threads = 8

        self.list_labels = range(0, 1)
        self.num_images_training = self.opt.dataset.num_images_training
        self.num_images_test = self.opt.dataset.num_images_testing

        self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.create_tfrecords()


    def get_parameters_complexity(self, complexity):
        if complexity == 0:
            num_points = [3,5]
            minimum_radius =[23,25]
            maximum_radius = [25,28]

        return num_points, maximum_radius, minimum_radius


    # Virtual functions:
    def get_data_trainval(self):

        num_points_range, maximum_radius_range, minimum_radius_range = \
            get_parameters_complexity(self.opt.dataset.complexity)

        # read the 5 batch files of cifar
        X = []
        labels = []
        for i in range(self.opt.dataset.num_images_training):
            num_points = np.randint(num_points_range[0],num_points_range[1])
            minimum_radius = np.randint(minimum_radius_range[0], minimum_radius_range[1])
            maximum_radius = np.randint(maximum_radius_range[0], maximum_radius_range[1])

            X.append(generate_dataset.generate_data(num_points, self.opt.dataset.image_size, self.opt.dataset.image_size,
                                   maximum_radius, minimum_radius))
            labels.append(X[-1].reshape(self.opt.dataset.num_images_training, self.opt.dataset.image_size**2))

        train_addrs = []
        train_labels = []
        val_addrs = []
        val_labels = []

        # Divide the data into 95% train, 5% validation
        [train_addrs.append(elem) for elem in X[0:int(self.opt.dataset.proportion_training_set * len(X))]]
        [train_labels.append(elem) for elem in labels[0:int(self.opt.dataset.proportion_training_set * len(X))]]

        [val_addrs.append(elem) for elem in X[int(self.opt.dataset.proportion_training_set * len(X)):]]
        [val_labels.append(elem) for elem in labels[int(self.opt.dataset.proportion_training_set * len(X))]]

        return train_addrs, train_labels, val_addrs, val_labels


    def get_data_test(self):
        # read the 5 batch files of cifar
        X = []
        labels = []
        for i in range(self.opt.dataset.num_images_training):
            X.append(get_new_data_point(self.opt.dataset.image_size, self.opt.dataset.complexity))
            labels.append(X[-1].reshape(self.opt.dataset.num_images_training, self.opt.dataset.image_size**2))

        return X, labels


    def preprocess_image(self, augmentation, standarization, image):

        return image
