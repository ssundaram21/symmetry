import numpy as np
import pickle

path = "/om/user/shobhita/data/symmetry/natural_images/"

with open(path + "symm_testing.pkl", "rb") as handle:
    full_test = np.asarray(pickle.load(handle))

for subset_no in range(10):
    subset = full_test[np.random.randint(len(full_test), size=500), :]
    with open(path + "symm_testing_s{}.pkl".format(subset_no), "wb") as handle:
        pickle.dump(subset, handle)