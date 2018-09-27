import numpy as np
import tensorflow as tf
from numba import jit
from skimage.morphology import binary_dilation

@jit(nopython=True)
def neighbours(point, shape):
    ne = []
    if point[0]>0:
        ne.append((point[0]-1, point[1]))
    if point[0]<shape[0]-1:
        ne.append((point[0]+1, point[1]))
    if point[1]>0:
        ne.append((point[0], point[1]-1))
    if point[1]<shape[1]-1:
        ne.append((point[0], point[1]+1))
    return ne

@jit(nopython=True)
def fillspace(img):
    filled = np.zeros_like(img)
    selected = [(0,0)]
    while len(selected) > 0:
        current = selected.pop()
        if not filled[current] and not img[current]:
            filled[current] = True
            selected.extend(neighbours(current, img.shape))
    return filled

def inner_border(image):
    return (binary_dilation(image)-image) * (1-fillspace(image))

def augment(data):
    """ Creates a new tensor with inner borders as the second channel

    :param data: input tensor (batch x h x w x 1)
    :return: augmented tensor (batch x h x w x 2)
    """
    with tf.Session() as sess:
        data_np = data.eval()

    inner = np.empty_like(data_np)
    for i in range(data_np.shape[0]):
        inner[i, :, :, 0] = inner_border(data_np[i, :, :, 0])
    return tf.concat([inner, data], 3)
