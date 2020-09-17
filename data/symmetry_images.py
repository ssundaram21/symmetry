import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# tf.enable_eager_execution()

classes = np.array(["0", "1"])
temp_path = "/om/user/shobhita/data/symmetry/raw_images/training/0/S0/"
print(temp_path)

img_files = os.listdir(temp_path)

test = img_files[0]

def get_label(path):
    parts = tf.strings.split(path, "/")
    one_hot = parts[-2] == classes
    return tf.argmax(one_hot)

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  return img



def get_image(path):
    with tf.Session() as sess:
        parts = path.split("/")
        print(parts)
        label = int(parts[-3])
        # load the raw data from the file as a string
        img = sess.run(tf.io.read_file(path))
        img = sess.run(tf.image.decode_png(img, channels=3))
        print("Image shape: ", img.shape)
        print(img)
        print(label)
        newimg = sess.run(tf.image.resize(img, [20,20]))
        print(newimg.shape)
        print(newimg)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        plt.imshow(img.astype("uint8"))
        plt.axis("off")


get_image(temp_path + test)
