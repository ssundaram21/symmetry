import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
# tf.enable_eager_execution()

LABELS = np.array(["0", "1"])
CATEGORY_LABEL_MAP = {"NS0": 0, "NS2": 0, "NS4": 0, "NS6": 0, "NSd4": 0, "S0": 1, "S2": 1, "S4": 1, "S6": 1, "Sd4": 1}
TRAIN_CATEGORIES = ["NS0", "NS4", "S0", "S4"]
CATS0 = ["NS0", "NS2", "NS4", "NS6", "NSd4"]
CATS1 = [ "S0", "S2", "S4", "S6", "Sd4"]
CATEGORIES = ["NS0", "NS2", "NS4", "NS6", "NSd4", "S0", "S2", "S4", "S6", "Sd4"]
TRAIN_SEPARATORS = np.array(["25", "50", "100", "150", "200", "1000"])
BASE_PATH = "/om/user/shobhita/data/symmetry/raw_images"
TARGET_PATH = "/om/user/shobhita/data/symmetry/np_images"
TARGET_DIM = 20


# Getting image paths
def get_cat_paths_test(base_path):
    paths = []
    for img_class in LABELS:
        categories = [x for x in os.listdir(os.path.join(base_path, img_class)) if not x.startswith('.')]
        paths += [os.path.join(base_path, img_class, c) for c in categories]
    return paths


def get_cat_paths_train(base_path):
    paths = []
    for directory in TRAIN_SEPARATORS:
        paths += [os.path.join(base_path, directory, x) for x in os.listdir(os.path.join(base_path, directory)) if not x.startswith('.')]
    return paths


def get_img_paths(cat_path):
     return [os.path.join(cat_path, img) for img in os.listdir(cat_path) if not img.startswith('.')]


def get_train_img_paths(base_path):
    train_path = os.path.join(base_path, "training")
    img_paths = []
    cat_paths = get_cat_paths_train(train_path)
    for cat in cat_paths:
        img_paths += get_img_paths(cat)
    return img_paths


def get_test_img_paths(base_path):
    test_path = os.path.join(base_path, "testing")
    img_paths = []
    cat_paths = get_cat_paths_test(test_path)
    for cat in cat_paths:
        img_paths += get_img_paths(cat)
    return img_paths



# Reading images
def get_label_test(img_path):
    path_split = np.array(img_path.split(os.sep))
    return int(path_split[np.isin(path_split, LABELS)][0])

def get_label_train(img_path):
    path_split = np.array(img_path.split(os.sep))
    cat = path_split[np.isin(path_split, CATEGORIES)][0]
    return CATEGORY_LABEL_MAP[cat]

def get_category(img_path):
    path_split = np.array(img_path.split(os.sep))
    cat = path_split[np.isin(path_split, CATEGORIES)][0]
    return cat

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    return img

def get_img(path, train=False):
#     print("\n\n IMAGE:", path)
    with tf.Session() as sess:
#         label = get_label_train(path) if train else get_label_test(path)
        category = get_category(path)
        # load the raw data from the file as a string
        img = sess.run(tf.io.read_file(path))
        img = sess.run(tf.image.decode_png(img))
    return img, category


def view_img(img):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    plt.imshow(img.astype("uint8"))
    plt.axis("off")



# Downscaling images
def finalize_repeats(repeats):
#     print(repeats)
    final = []
    for x in repeats:
        if x > 13:
            final += [11 for i in range(int(np.round(x / 11)))]
        else:
            final.append(x)
#     print(final)
    cumsum = np.array(final).cumsum()
    cumsum[-1] = 224
    return cumsum


def get_horizontal_repeats(img):
#     print("H")
    repeats = []
    i = 0
    curr_val = img[0,0,0]
    for j in range(len(img)):
        if img[0,j,0] != curr_val:
            curr_val = img[0,j,0]
            repeats.append(i)
            i = 1
        else:
            i += 1
    repeats.append(i)
    return finalize_repeats(repeats)


def get_vertical_repeats(img):
#     print("V")
    repeats = []
    i = 0
    curr_val = img[0,0,0]
    for j in range(len(img)):
        if img[j,0,0] != curr_val:
            curr_val = img[j,0,0]
            repeats.append(i)
            i = 1
        else:
            i += 1
    repeats.append(i)
    return finalize_repeats(repeats)


def downscale_img(img, dim):
    h = get_vertical_repeats(img)
#     print(h)
    v = get_horizontal_repeats(img)
#     print(v)
    ds_img = np.zeros((dim, dim))
    curr = img[0,0,0]
    for i in range(dim):
        for j in range(dim):
            col = h[i]-1
            row = v[j]-1
            ds_img[i, j] = img[col,row,0]

    return ds_img



# Wrapper
def process_train_data(base_path, target_path):
    result_path = os.path.join(target_path, "training")

    img_paths = get_train_img_paths(base_path)
    downscaled_imgs = {cat: [] for cat in CATEGORIES}

    count = 0
    file_no = 0
    n = len(img_paths)
    for img_path in img_paths:
        img, cat = get_img(img_path, train=True)
        img_ds = downscale_img(img, TARGET_DIM)
        downscaled_imgs[cat].append(img_ds)
        count += 1

        if count % 200 == 0:
            print("train img {} / {}".format(count, n))

    print("Saving test images")
    for cat in TRAIN_CATEGORIES:
        with open(os.path.join(result_path, "{}.pkl".format(cat)), "wb+") as file:
            pickle.dump(np.array(downscale_imgs[cat], file))


def process_test_data(base_path, target_path):
    result_path = os.path.join(target_path, "testing")

    img_paths = get_test_img_paths(base_path)
    downscaled_imgs = {cat: [] for cat in CATEGORIES}

    count = 0
    file_no = 0
    n = len(img_paths)
    for img_path in img_paths:
        img, cat = get_img(img_path, train=False)
        img_ds = downscale_img(img, TARGET_DIM)
        downscaled_imgs[cat].append(img_ds)
        count += 1

        if count % 200 == 0:
            print("test img {} / {}".format(count, n))

    print("Saving test images")
    for cat in CATEGORIES:
        with open(os.path.join(result_path, "{}.pkl".format(cat)), "wb+") as file:
            pickle.dump(np.array(downscale_imgs[cat], file))

def run():
    print("Processing training data...")
    process_train_data(BASE_PATH, TARGET_PATH)

    print("Processing testing data...")
    process_test_data(BASE_PATH, TARGET_PATH)

    print("done :)")

def get_np_data():
    subfolder = "training"
    path = os.listdir(os.path.join(target_path, test_subfolder))
    imgs = []
    labels = []

    for cat in CATS0:
        if cat in TRAIN_CATEGORIES:
            file = np.array(os.path.join(path, ))


    file = np.array(os.listdir(os.path.join(target_path, test_subfolder, "{}.pkl".format(category))))
    with open(file, "rb") as cat_file:
        data = pickle.load(file)
    return data

if __name__ == '__main__':
    run()