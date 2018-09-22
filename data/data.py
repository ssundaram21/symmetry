import tensorflow as tf
import sys
import os.path
import shutil

class Dataset:

    num_threads = 8

    num_outputs = 0
    list_labels = range(0)
    num_images_training = 0
    num_images_val = 0
    num_images_test = 0

    def __init__(self, opt):
        self.opt = opt

    #ABSTRACT METHOD
    def get_data_trainval(self):
        # Returns images training & labels
        pass

    #ABSTRACT METHOD
    def get_data_test(self):
        # Returns images training & labels
        pass

    #ABSTRACT METHOD
    def preprocess_image(self, augmentation, standarization, image, labels):
        # Returns images training & labels
        pass

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Write one TF records file
    def write_tfrecords(self, tfrecords_path, set_name, addrs, labels, img_size, addrs_raw):

        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(tfrecords_path + set_name + '.tfrecords')

        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 10:
                print('Data: {}/{}'.format(i, len(addrs)))
                sys.stdout.flush()

            # Create a feature
            feature = {set_name + '/label': self._bytes_feature(labels[i].tostring()),
                       set_name + '/image': self._bytes_feature(addrs[i].tostring()),
                       set_name + '/image_raw': self._bytes_feature(addrs_raw[i].tostring()),
                       set_name + '/width': self._int64_feature(img_size),
                       set_name + '/height': self._int64_feature(img_size)
                       }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    # Create all TFrecords files
    def create_tfrecords(self):

        tfrecords_path = self.opt.log_dir_base + self.opt.dataset.log_name + '/data/'

        if os.path.isfile(tfrecords_path + 'test.tfrecords'):
            print("REUSING TFRECORDS")
            sys.stdout.flush()
            return 0

        if os.path.isdir(tfrecords_path):
            shutil.rmtree(tfrecords_path)

        os.makedirs(tfrecords_path)

        print("CREATING TFRECORDS")
        sys.stdout.flush()
        print(self.opt.dataset.dataset_path)

        train_addrs, train_labels, train_addrs_raw, val_addrs, val_labels, val_addrs_raw = self.get_data_trainval()
        self.write_tfrecords(tfrecords_path, 'train', train_addrs, train_labels, self.opt.dataset.image_size, train_addrs_raw)
        self.write_tfrecords(tfrecords_path, 'val', val_addrs, val_labels, self.opt.dataset.image_size,val_addrs_raw)

        test_addrs, test_labels, test_addrs_raw = self.get_data_test()
        self.write_tfrecords(tfrecords_path, 'test', test_addrs, test_labels, self.opt.dataset.image_size, test_addrs_raw)


    # Create all TFrecords files
    def create_tfrecords_from_numpy(self, data):

        tfrecords_path = self.opt.log_dir_base + self.opt.dataset.log_name + '/data/'

        if os.path.isfile(tfrecords_path + 'test.tfrecords'):
            print("OVERWRITTING TFRECORDS")
            sys.stdout.flush()

        print("CREATING TFRECORDS")
        sys.stdout.flush()
        print(self.opt.dataset.dataset_path)

        train_addrs = data['train_img']; train_labels = data['train_gt']
        val_addrs = data['val_img']; val_labels = data['val_gt']
        self.write_tfrecords(tfrecords_path, 'train', train_addrs, train_labels, self.opt.dataset.image_size)
        self.write_tfrecords(tfrecords_path, 'val', val_addrs, val_labels, self.opt.dataset.image_size)

        test_addrs = data['test_img']; test_labels = data['test_gt']
        self.write_tfrecords(tfrecords_path, 'test', test_addrs, test_labels, self.opt.dataset.image_size)

    def create_dataset(self, augmentation=False, standarization=False, set_name='train', repeat=False):
        set_name_app = set_name

        # Transforms a scalar string `example_proto` into a pair of a scalar string and
        # a scalar integer, representing an image and its label, respectively.
        def _parse_function(example_proto):
            features = {set_name_app + '/label': tf.FixedLenFeature((), tf.string, default_value=""),
                        set_name_app + '/image': tf.FixedLenFeature((), tf.string, default_value=""),
                        set_name_app + '/image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                        set_name_app + '/height': tf.FixedLenFeature([], tf.int64),
                        set_name_app + '/width': tf.FixedLenFeature([], tf.int64)}

            parsed_features = tf.parse_single_example(example_proto, features)

            image = tf.decode_raw(parsed_features[set_name_app + '/image'], tf.uint8)
            image = tf.cast(image, tf.float32)
            S = tf.stack([tf.cast(parsed_features[set_name_app + '/height'], tf.int32),
                          tf.cast(parsed_features[set_name_app + '/width'], tf.int32)])
            image = tf.reshape(image, S)

            image_raw = tf.decode_raw(parsed_features[set_name_app + '/image_raw'], tf.uint8)
            image_raw = tf.cast(image_raw, tf.float32)
            S = tf.stack([tf.cast(parsed_features[set_name_app + '/height'], tf.int32),
                          tf.cast(parsed_features[set_name_app + '/width'], tf.int32)])
            image_raw = tf.reshape(image_raw, S)

            label = tf.decode_raw(parsed_features[set_name_app + '/label'], tf.uint8)
            label = tf.cast(label, tf.float32)
            S = tf.stack([tf.cast(parsed_features[set_name_app + '/height'], tf.int32),
                          tf.cast(parsed_features[set_name_app + '/width'], tf.int32)])
            label = tf.reshape(label, S)
            label = tf.cast(label, tf.int64)

            float_image, float_labels, float_raw = self.preprocess_image(augmentation, standarization, image, label, image_raw)
            return float_image, label, float_raw

        tfrecords_path = self.opt.log_dir_base + self.opt.dataset.log_name + '/data/'

        filenames = [tfrecords_path + set_name_app + '.tfrecords']
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_parallel_calls=self.num_threads)

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely.

        return dataset.batch(self.opt.hyper.batch_size)
