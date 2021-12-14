import numpy as np
import pickle

path = "/om/user/shobhita/data/symmetry/natural_images/"

def get_natural_testing_im():
    return pickle.load(open(path + "symm_testing.pkl", "rb"))

images = get_natural_testing_im()
mirrored_images = []
for im, label in images:
    if label == 1:
        left_flank = im[:, :len(im) // 2]
        right_flank = np.flip(left_flank, axis=1)
        im = np.concatenate((left_flank, right_flank), axis=1)
    mirrored_images.append((im, label))
pickle.dump(np.asarray(mirrored_images), open(path + "mirrored_symm_testing.pkl", "wb"))