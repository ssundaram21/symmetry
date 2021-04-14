import numpy as np
from random import randint

def NS0(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.random.randint(0, 256, (imsize, imsize // 2))
    return np.concatenate([left, right], axis=1), 0

def NS2(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 1))
    right = np.random.randint(0, 256, (imsize, imsize // 2 - 1))
    mid = np.full((imsize, 2), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def NS4(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 2))
    right = np.random.randint(0, 256, (imsize, imsize // 2 - 2))
    mid = np.full((imsize, 4), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def NS6(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 3))
    right = np.random.randint(0, 256, (imsize, imsize // 2 - 3))
    mid = np.full((imsize, 6), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def NSd4(imsize):
    left = np.random.randint(0, 129, (imsize, imsize // 2 - 2))
    right = np.random.randint(0, 129, (imsize, imsize // 2 - 2))
    mid = np.full((imsize, 4), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def S0(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.flip(left, axis=1)
    return np.concatenate([left, right], axis=1), 1

def S2(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 1))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, 2), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def S4(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 2))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, 4), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def S6(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2 - 3))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, 6), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def Sd4(imsize):
    left = np.random.randint(0, 129, (imsize, imsize // 2 - 2))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, 4), 128)
    return np.concatenate([left, mid, right], axis=1), 1

IMAGE_TYPE = {
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

def make_images(imtype, imsize = 20, n_images=1):
    images = []
    labels = []
    while len(images) < n_images:
        image, label = IMAGE_TYPE[imtype](imsize)
        images.append(image)
        labels.append(label)

    return images, labels

def make_random(imtypes, imsize=20, n_images=1):
    images = []
    labels = []
    while len(images) < n_images:
        imtype = imtypes[randint(0, len(imtypes)-1)]
        image, label = IMAGE_TYPE[imtype](imsize)
        images.append(image)
        labels.append(label)
    return images, labels

