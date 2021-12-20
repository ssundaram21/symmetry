import numpy as np
from random import randint

################# ORIGINAL DATASETS
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


################## FLANK DATASETS
def flank1S(imsize):
    left = np.random.randint(0, 256, (imsize, 1))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, imsize - 2), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def flank2S(imsize):
    left = np.random.randint(0, 256, (imsize, 2))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, imsize - 4), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def flank3S(imsize):
    left = np.random.randint(0, 256, (imsize, 3))
    right = np.flip(left, axis=1)
    mid = np.full((imsize, imsize - 6), 128)
    return np.concatenate([left, mid, right], axis=1), 1

def flank1NS(imsize):
    left = np.random.randint(0, 256, (imsize, 1))
    right = np.random.randint(0, 256, (imsize, 1))
    mid = np.full((imsize, imsize - 2), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def flank2NS(imsize):
    left = np.random.randint(0, 256, (imsize, 2))
    right = np.random.randint(0, 256, (imsize, 2))
    mid = np.full((imsize, imsize - 4), 128)
    return np.concatenate([left, mid, right], axis=1), 0

def flank3NS(imsize):
    left = np.random.randint(0, 256, (imsize, 3))
    right = np.random.randint(0, 256, (imsize, 3))
    mid = np.full((imsize, imsize - 6), 128)
    return np.concatenate([left, mid, right], axis=1), 0

######################### STRIPE DATASETS
def stripe2S(imsize):
    stripe = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[1,18]] = stripe
    return im, 1

def stripe4S(imsize):
    stripe = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[3,16]] = stripe
    return im, 1

def stripe6S(imsize):
    stripe = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[5,14]] = stripe
    return im, 1

def stripe8S(imsize):
    stripe = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[7,12]] = stripe
    return im, 1

def stripe10S(imsize):
    stripe = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[9,10]] = stripe
    return im, 1

def stripe2NS(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[1]] = stripe_left
    im[:,[18]] = stripe_right
    return im, 0

def stripe4NS(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[3]] = stripe_left
    im[:,[16]] = stripe_right
    return im, 0

def stripe6NS(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[5]] = stripe_left
    im[:,[14]] = stripe_right
    return im, 0

def stripe8NS(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[7]] = stripe_left
    im[:,[12]] = stripe_right
    return im, 0

def stripe10NS(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[9]] = stripe_left
    im[:,[10]] = stripe_right
    return im, 0


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
    "Sd4": Sd4,
    "flank1S": flank1S,
    "flank2S": flank2S,
    "flank3S": flank3S,
    "flank1NS": flank1NS,
    "flank2NS": flank2NS,
    "flank3NS": flank3NS,
    "stripe2S": stripe2S,
    "stripe4S": stripe4S,
    "stripe6S": stripe6S,
    "stripe8S": stripe8S,
    "stripe10S": stripe10S,
    "stripe2NS": stripe2NS,
    "stripe4NS": stripe4NS,
    "stripe6NS": stripe6NS,
    "stripe8NS": stripe8NS,
    "stripe10NS": stripe10NS,
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

