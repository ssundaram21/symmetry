import numpy as np
import pickle

def get_natural_training_im(raw_data_path, im_index):
    symm_images = pickle.load(open(raw_data_path + "symm_training.pkl", "rb"))
    return symm_images[im_index]

def get_natural_testing_im(raw_data_path, im_index):
    nonsymm_images = pickle.load(open(raw_data_path + "symm_testing.pkl", "rb"))
    return nonsymm_images[im_index]

def get_natural_testing_im_mirror(im_index):
    im, label = get_natural_testing_im(im_index)
    if label == 1:
        left_flank = im[:, :len(im) // 2]
        right_flank = np.flip(left_flank, axis=1)
        im = np.concatenate((left_flank, right_flank), axis=1)
    return im, label

IM_TYPE = {
    "natTrain": get_natural_training_im,
    "natTest": get_natural_testing_im,
    "natTestMirror": get_natural_testing_im_mirror}

def get_natural_image(type, im_index, raw_data_path, n = 1):
    images = []
    labels = []
    while len(images) < n:
        image, label = IM_TYPE[type](raw_data_path, im_index)
        images.append(image)
        labels.append(label)
    return images, labels
