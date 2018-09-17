import tensorflow as tf
import sys
import os.path
import shutil

from abc import ABCMeta, abstractmethod


class Dataset:

    __metaclass__ = ABCMeta

    num_threads = 8
    output_buffer_size = 1024

    list_labels = range(0)
    num_images_training = 0
    num_images_test = 0

    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def get_data_trainval(self):
        # Returns images training & labels
        pass

    @abstractmethod
    def get_data_test(self):
        # Returns images training & labels
        pass

    @abstractmethod
    def preprocess_image(self, image):
        # Returns images training & labels
        pass

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Write one TF records file
    def write_tfrecords(self, tfrecords_path, set_name, addrs, labels):

        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(tfrecords_path + set_name + '.tfrecords')

        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print('Data: {}/{}'.format(i, len(addrs)))
                sys.stdout.flush()

            # Create a feature
            feature = {set_name + '/label': self._int64_feature(labels[i]),
                       set_name + '/image': self._bytes_feature(addrs[i].tostring()),
                       set_name + '/width': self._int64_feature(32),
                       set_name + '/height': self._int64_feature(32)
                       }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    # Create all TFrecords files
    def create_tfrecords(self):

        if not self.opt.dataset.reuse_TFrecords:
            tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        else:
            tfrecords_path = self.opt.log_dir_base + self.opt.dataset.reuse_TFrecords_path + '/data/'
            print("REUSING TFRECORDS")

        if os.path.isfile(tfrecords_path + 'test.tfrecords'):
            return 0

        if os.path.isdir(tfrecords_path):
            shutil.rmtree(tfrecords_path)

        os.makedirs(tfrecords_path)

        print("CREATING TFRECORDS")
        print(self.opt.dataset.dataset_path)

        train_addrs, train_labels, val_addrs, val_labels = self.get_data_trainval()
        app = self.opt.dataset.transfer_append_name
        self.write_tfrecords(tfrecords_path, 'train' + app, train_addrs, train_labels)
        self.write_tfrecords(tfrecords_path, 'val' + app, val_addrs, val_labels)

        test_addrs, test_labels = self.get_data_test()
        self.write_tfrecords(tfrecords_path, 'test' + app, test_addrs, test_labels)

    def delete_tfrecords(self):
        tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        shutil.rmtree(tfrecords_path)

    def create_dataset(self, augmentation=False, standarization=False, set_name='train', repeat=False):
        app = self.opt.dataset.transfer_append_name
        set_name_app = set_name + app

        # Transforms a scalar string `example_proto` into a pair of a scalar string and
        # a scalar integer, representing an image and its label, respectively.
        def _parse_function(example_proto):
            features = {set_name_app + '/label': tf.FixedLenFeature((), tf.int64, default_value=1),
                        set_name_app + '/image': tf.FixedLenFeature((), tf.string, default_value=""),
                        set_name_app + '/height': tf.FixedLenFeature([], tf.int64),
                        set_name_app + '/width': tf.FixedLenFeature([], tf.int64)}
            parsed_features = tf.parse_single_example(example_proto, features)
            image = tf.decode_raw(parsed_features[set_name_app + '/image'], tf.uint8)
            image = tf.cast(image,tf.float32)
            S = tf.stack([tf.cast(parsed_features[set_name_app + '/height'], tf.int32),
                          tf.cast(parsed_features[set_name_app + '/width'], tf.int32), 3])
            image = tf.reshape(image, S)

            float_image = self.preprocess_image(augmentation, standarization, image)

            return float_image, parsed_features[set_name_app + '/label']

        # Creates a dataset that reads all of the examples from two files, and extracts
        # the image and label features.
        if not self.opt.dataset.reuse_TFrecords:
            tfrecords_path = self.opt.log_dir_base + self.opt.name + '/data/'
        else:
            tfrecords_path = self.opt.log_dir_base + self.opt.dataset.reuse_TFrecords_path + '/data/'

        filenames = [tfrecords_path + set_name_app + '.tfrecords']
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_threads=self.num_threads, output_buffer_size=self.output_buffer_size)

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely.

        return dataset.batch(self.opt.hyper.batch_size)
