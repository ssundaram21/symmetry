import numpy as np
import pickle

path = "/om/user/shobhita/data/symmetry/natural_images/"

def get_symmetric_im(im_index):
    symm_images = pickle.load(open(path + "symmetric_images_unmirrored.pkl", "rb"))
    return symm_images[im_index]

def get_symmetric2_im(im_index):
    symm_images = pickle.load(open(path + "gray_cropped_imaged_unmirrored.pkl", "rb"))
    return symm_images[im_index]

def get_nonsymmetric_im(im_index):
    nonsymm_images = pickle.load(open(path + "nonsymmetric_images.pkl", "rb"))
    return nonsymm_images[im_index]


def get_natural_training_im(im_index):
    symm_images = pickle.load(open(path + "symm_training.pkl", "rb"))
    return symm_images[im_index]

def get_natural_training_im_100(im_index):
    symm_images = pickle.load(open(path + "symm_training_100.pkl", "rb"))
    return symm_images[im_index]

def get_natural_training_im_1000(im_index):
    symm_images = pickle.load(open(path + "symm_training_1000.pkl", "rb"))
    return symm_images[im_index]

def get_natural_training_im_10000(im_index):
    symm_images = pickle.load(open(path + "symm_training_10000.pkl", "rb"))
    return symm_images[im_index]

def get_natural_testing_im(im_index):
    nonsymm_images = pickle.load(open(path + "symm_testing.pkl", "rb"))
    return nonsymm_images[im_index]

def get_natural_testing_im_mirror(im_index):
    im, label = get_natural_testing_im(im_index)
    if label == 1:
        left_flank = im[:, :len(im) // 2]
        right_flank = np.flip(left_flank, axis=1)
        im = np.concatenate((left_flank, right_flank), axis=1)
    return im, label

def get_natural_testing_subset(im_index, subset_no):
    images = pickle.load(open(path + f"symm_testing_s{subset_no}.pkl", "rb"))
    return images[im_index]

def get_natural_testing_subset_mirror(im_index, subset_no):
    im, label = get_natural_testing_subset(im_index, subset_no)
    if label == 1:
        left_flank = im[:, :len(im) // 2]
        right_flank = np.flip(left_flank, axis=1)
        im = np.concatenate((left_flank, right_flank), axis=1)
    return im, label


IM_TYPE = {
    "natS": get_symmetric_im,
    "natS2": get_symmetric2_im,
    "natNS": get_nonsymmetric_im,
    "natTrain": get_natural_training_im,
    "natTrain100": get_natural_training_im_100,
    "natTrain1000": get_natural_training_im_1000,
    "natTrain10000": get_natural_training_im_10000,
    "natTest": get_natural_testing_im,
    "natTestMirror": get_natural_testing_im_mirror}

SUBSET_IM_TYPE = {
    "natTestSubset": get_natural_testing_subset,
    "natTestMirrorSubset": get_natural_testing_subset_mirror
}


def get_natural_image(type, im_index, n = 1):
    images = []
    labels = []
    while len(images) < n:
        image, label = IM_TYPE[type](im_index)
        images.append(image)
        labels.append(label)
    return images, labels

def get_natural_image_subset(type, subset_no, im_index, n = 1):
    images = []
    labels = []
    while len(images) < n:
        image, label = SUBSET_IM_TYPE[type](im_index, subset_no)
        images.append(image)
        labels.append(label)
    return images, labels