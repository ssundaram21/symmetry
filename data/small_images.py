import numpy as np
from numba import jit
from random import randint, seed
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from collections import namedtuple

directions = ((1, 0), (0, -1), (-1, 0), (0, 1))

ImageSet = namedtuple("ImageSet", "curve inside start")


@jit(nopython=True)
def get_dir(i_dir):
    return directions[(i_dir) % 4]


@jit(nopython=True)
def add_vec(v0, v1):
    return (v0[0] + v1[0], v0[1] + v1[1])


@jit(nopython=True)
def in_bounds(point, image):
    return 0 < point[0] < image.shape[0] - 1 and 0 < point[1] < image.shape[1] - 1


@jit(nopython=True)
def spaced(point, i_dir, image):
    pf = add_vec(point, get_dir(i_dir))
    pl = add_vec(pf, get_dir(i_dir + 1))
    pr = add_vec(pf, get_dir(i_dir - 1))
    if in_bounds(pl, image) and image[pl]:
        return False
    if in_bounds(pr, image) and image[pr]:
        return False
    return True


@jit(nopython=True)
def eligible(point, i_dir, image):
    dp = add_vec(point, get_dir(i_dir))
    cond = in_bounds(dp, image) and spaced(point, i_dir, image) and image[dp] == 0
    return cond


# Debugging functions
string_dirs = "↓←↑→"


@jit(nopython=True)
def string_dir(i_dir):
    return string_dirs[(i_dir) % 4]


@jit(nopython=True)
def draw_img_ascii(img, current, i_dir):
    lines = ""
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i, j) == current:
                lines += string_dir(i_dir)
            elif img[i, j] > 0:
                lines += "#"
            else:
                lines += "·"
        lines += "\n"
    return lines


@jit(nopython=True)
def draw_history(explore):
    exp = np.zeros((4, len(explore)), dtype=np.uint8)
    lines = ""
    for i, ei in enumerate(explore):
        exp[0, i] = ei[1]
        exp[1:4, i] = ei[2]
    for i in range(4):
        for j in range(len(explore)):
            if i == 0:
                lines += string_dir(exp[i, j])
            else:
                lines += "#" if exp[i, j] else "·"
        lines += "\n"
    return lines


@jit(nopython=True)
def create_loop(w, h, max_iterations=1000):
    """ Creates a closed curve by random exploration with backtracking
    :param w:
    :param h:
    :param max_iterations:
    :return:
    """
    start_x = randint(1, w - 2)
    start_y = randint(1, h - 2)
    image = np.zeros((w, h), dtype=np.uint8)
    start = (start_y, start_x)
    current = start
    explore = []
    n_total = 0
    i_dir = randint(0, 3)
    explore.append((current, i_dir, np.array([1, 1, 1], dtype=np.uint8)))
    c_len = 1
    success = False
    while len(explore) > 0 and n_total < max_iterations:
        n_total += 1
        point, i_dir, next_remaining = explore[-1]
        # if the curve is closed, our job is done:
        if c_len > 3 and add_vec(point, get_dir(i_dir)) == start:
            next_point = add_vec(point, get_dir(i_dir))
            image[point] = 1
            success = True
            break

        # if we are adding a new point, check if it is eligible:
        if eligible(point, i_dir, image):
            next_point = add_vec(point, get_dir(i_dir))
            image[point] = 1
            next_dir = randint(-1, 1)
            all_next = np.ones(3, np.uint8)
            all_next[next_dir + 1] = 0
            c_len += 1
            explore.append((next_point, i_dir + next_dir, all_next))
        else:
            while np.sum(next_remaining) == 0:
                if len(explore) == 0:
                    return explore, image, False
                else:
                    point, i_dir, next_remaining = explore.pop()
                    image[point] = 0
            remaining_choice = np.random.choice(np.nonzero(next_remaining)[0])
            next_remaining[remaining_choice] = 0
            explore.append((point, i_dir + remaining_choice - 1, next_remaining))

    return explore, image, success


def make_imset(im):
    filled = binary_fill_holes(im)
    inside = filled - im
    all_inside = np.nonzero(inside)
    some_inside = np.random.choice(len(all_inside[0]))
    hint = np.zeros_like(inside)
    hint[all_inside[0][some_inside], all_inside[1][some_inside]] = 1
    return inside, hint


@jit(nopython=True)
def set_seed(i_seed):
    print("Set numba seed")
    seed(i_seed)


def make_small_images(im_size=(8, 8), n_images=4, rejection_function=None):
    images = []
    if rejection_function is None:

        def rejection_function(x):
            return np.sum(x) < 2

    while len(images) < n_images:
        pth, im, suc = create_loop(*im_size)
        if suc:
            inside, hint = make_imset(im)
            lbl, num = label(inside, connectivity=1, return_num=True)
            if num == 1 and not rejection_function(inside):
                images.append(ImageSet(im, inside, hint))
    return images

