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



############### DIFF DATASETS
def diff1NS(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.flip(left, axis=1)
    im = np.concatenate([left, right], axis=1)
    change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
    old_val = im[change_x][change_y]
    new_val = np.random.randint(0, 256)
    while new_val == old_val:
        new_val = np.random.randint(0, 256)
    im[change_x, change_y] = new_val
    return im, 0

def diff3NS(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.flip(left, axis=1)
    im = np.concatenate([left, right], axis=1)
    changed_pixels = set()
    for _ in range(3):
        change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        while (change_x, change_y) in changed_pixels:
            change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        changed_pixels.add((change_x, change_y))
        old_val = im[change_x][change_y]
        new_val = np.random.randint(0, 256)
        while new_val == old_val:
            new_val = np.random.randint(0, 256)
        im[change_x, change_y] = new_val
    return im, 0

def diff5NS(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.flip(left, axis=1)
    im = np.concatenate([left, right], axis=1)
    changed_pixels = set()
    for _ in range(5):
        change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        while (change_x, change_y) in changed_pixels:
            change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        changed_pixels.add((change_x, change_y))
        old_val = im[change_x][change_y]
        new_val = np.random.randint(0, 256)
        while new_val == old_val:
            new_val = np.random.randint(0, 256)
        im[change_x, change_y] = new_val
    return im, 0

def diff10NS(imsize):
    left = np.random.randint(0, 256, (imsize, imsize // 2))
    right = np.flip(left, axis=1)
    im = np.concatenate([left, right], axis=1)
    changed_pixels = set()
    for _ in range(10):
        change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        while (change_x, change_y) in changed_pixels:
            change_x, change_y = np.random.randint(0, 20), np.random.randint(0, 20)
        changed_pixels.add((change_x, change_y))
        old_val = im[change_x][change_y]
        new_val = np.random.randint(0, 256)
        while new_val == old_val:
            new_val = np.random.randint(0, 256)
        im[change_x, change_y] = new_val
    return im, 0


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

######################### FURTHER TEST DATASETS
def test96(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 1))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 1))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[2]] = stripe_symm
    im[:,[17]] = stripe_symm
    im[:, [7]] = stripe_ns_left
    im[:, [12]] = stripe_ns_right
    return im, 0

def test97(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 1))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 1))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 1))
    im = np.full((imsize, imsize), 128)
    im[:,[7]] = stripe_symm
    im[:,[12]] = stripe_symm
    im[:, [2]] = stripe_ns_left
    im[:, [17]] = stripe_ns_right
    return im, 0

def test98(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 6))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 4))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 4))
    im = np.full((imsize, imsize), 128)
    im[:,:4] = stripe_ns_left
    im[:,4:10] = stripe_symm
    im[:,10:16] = np.flip(stripe_symm, axis=1)
    im[:, 16:] = stripe_ns_right
    return im, 0

def test99(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 4))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 6))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 6))
    im = np.full((imsize, imsize), 128)
    im[:,:4] = stripe_symm
    im[:,4:10] = stripe_ns_left
    im[:,10:16] = stripe_ns_right
    im[:, 16:] = np.flip(stripe_symm, axis=1)
    return im, 0

def test100(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 4))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,:4] = stripe_symm
    im[:,4:7] = stripe_ns_left
    im[:,13:16] = stripe_ns_right
    im[:, 16:] = np.flip(stripe_symm, axis=1)
    return im, 0

def test101(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 4))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,:3] = stripe_ns_left
    im[:,3:7] = stripe_symm
    im[:,13:17] = np.flip(stripe_symm, axis=1)
    im[:, 17:] = stripe_ns_right
    return im, 0

def test102(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,:3] = stripe_symm
    im[:,5:8] = stripe_ns_left
    im[:,12:15] = stripe_ns_right
    im[:, 17:] = np.flip(stripe_symm, axis=1)
    return im, 0

def test103(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,:3] = stripe_ns_left
    im[:,5:8] = stripe_symm
    im[:,12:15] = np.flip(stripe_symm, axis=1)
    im[:, 17:] = stripe_ns_right
    return im, 0

def test104(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 2))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,2:4] = stripe_symm
    im[:,6:9] = stripe_ns_left
    im[:,11:14] = stripe_ns_right
    im[:, 16:18] = np.flip(stripe_symm, axis=1)
    return im, 0

def test105(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 2))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 3))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 3))
    im = np.full((imsize, imsize), 128)
    im[:,2:5] = stripe_ns_left
    im[:,7:9] = stripe_symm
    im[:,11:13] = np.flip(stripe_symm, axis=1)
    im[:, 15:18] = stripe_ns_right
    return im, 0

def test106(imsize):
    stripe_symm = np.random.randint(0, 256, (imsize, 1))
    stripe_ns_left = np.random.randint(0, 256, (imsize, 4))
    stripe_ns_right = np.random.randint(0, 256, (imsize, 4))
    im = np.full((imsize, imsize), 128)
    im[:,:4] = stripe_ns_left
    im[:,4:10] = stripe_symm
    im[:,10:16] = np.flip(stripe_symm, axis=1)
    im[:, 16:] = stripe_ns_right
    return im, 0


################ FURTHER TRAINING DATASETS
def NS4_filled(imsize):
    left = np.random.randint(0, 256, (imsize, imsize//2-2))
    right = np.random.randint(0, 256, (imsize, imsize//2-2))
    mid_symm = np.random.randint(0, 256, (imsize, 2))
    return np.concatenate([left, mid_symm, np.flip(mid_symm, axis=1), right], axis=1), 0

def NS8_filled(imsize):
    left = np.random.randint(0, 256, (imsize, 6))
    right = np.random.randint(0, 256, (imsize, 6))
    mid_symm = np.random.randint(0, 256, (imsize, 4))
    return np.concatenate([left, mid_symm, np.flip(mid_symm, axis=1), right], axis=1), 0

def NS12_filled(imsize):
    left = np.random.randint(0, 256, (imsize, 4))
    right = np.random.randint(0, 256, (imsize, 4))
    mid_symm = np.random.randint(0, 256, (imsize, 6))
    return np.concatenate([left, mid_symm, np.flip(mid_symm, axis=1), right], axis=1), 0

def NS16_filled(imsize):
    left = np.random.randint(0, 256, (imsize, 2))
    right = np.random.randint(0, 256, (imsize, 2))
    mid_symm = np.random.randint(0, 256, (imsize, 8))
    return np.concatenate([left, mid_symm, np.flip(mid_symm, axis=1), right], axis=1), 0

def NS18_filled(imsize):
    left = np.random.randint(0, 256, (imsize, 1))
    right = np.random.randint(0, 256, (imsize, 1))
    mid_symm = np.random.randint(0, 256, (imsize, 9))
    return np.concatenate([left, mid_symm, np.flip(mid_symm, axis=1), right], axis=1), 0


def NS2_stripe_fill(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 2))
    stripe_right = np.random.randint(0, 256, (imsize, 2))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,:2] = stripe_left
    im[:,18:] = stripe_right
    return im, 0

def NS4_stripe_fill(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 2))
    stripe_right = np.random.randint(0, 256, (imsize, 2))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,2:4] = stripe_left
    im[:,16:18] = stripe_right
    return im, 0

def NS6_stripe_fill(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 2))
    stripe_right = np.random.randint(0, 256, (imsize, 2))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,4:6] = stripe_left
    im[:,14:16] = stripe_right
    return im, 0

def NS8_stripe_fill(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 2))
    stripe_right = np.random.randint(0, 256, (imsize, 2))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,6:8] = stripe_left
    im[:,12:14] = stripe_right
    return im, 0

def NS10_stripe_fill(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 2))
    stripe_right = np.random.randint(0, 256, (imsize, 2))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,8:10] = stripe_left
    im[:,10:12] = stripe_right
    return im, 0

def NS2_stripe_fill1(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,[1]] = stripe_left
    im[:,[18]] = stripe_right
    return im, 0

def NS4_stripe_fill1(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,[3]] = stripe_left
    im[:,[16]] = stripe_right
    return im, 0

def NS6_stripe_fill1(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,[5]] = stripe_left
    im[:,[14]] = stripe_right
    return im, 0

def NS8_stripe_fill1(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
    im[:,[7]] = stripe_left
    im[:,[12]] = stripe_right
    return im, 0

def NS10_stripe_fill1(imsize):
    stripe_left = np.random.randint(0, 256, (imsize, 1))
    stripe_right = np.random.randint(0, 256, (imsize, 1))
    im_half = np.random.randint(0, 256, (imsize, imsize//2))
    im = np.concatenate([im_half, np.flip(im_half, axis=1)], axis=1)
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
    "diff1NS": diff1NS,
    "diff3NS": diff3NS,
    "diff5NS": diff5NS,
    "diff10NS": diff10NS,
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
    "test96": test96,
    "test97": test97,
    "test98": test98,
    "test99": test99,
    "test100": test100,
    "test101": test101,
    "test102": test102,
    "test103": test103,
    "test104": test104,
    "test105": test105,
    "test106": test106,
    "NS4Fill": NS4_filled,
    "NS8Fill": NS8_filled,
    "NS12Fill": NS12_filled,
    "NS16Fill": NS16_filled,
    "NS18Fill": NS18_filled,
    "NS2FillStripe": NS2_stripe_fill,
    "NS4FillStripe": NS4_stripe_fill,
    "NS6FillStripe": NS6_stripe_fill,
    "NS8FillStripe": NS8_stripe_fill,
    "NS10FillStripe": NS10_stripe_fill1,
    "NS2FillStripe1": NS2_stripe_fill1,
    "NS4FillStripe1": NS4_stripe_fill1,
    "NS6FillStripe1": NS6_stripe_fill1,
    "NS8FillStripe1": NS8_stripe_fill1,
    "NS10FillStripe1": NS10_stripe_fill1,
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

