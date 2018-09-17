import tensorflow as tf


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._jpeg_data, channels=3)

        # Initializes function that encodes RGB JPEG data.
        self._image = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
        self._image, format='rgb', quality=100)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        return self._sess.run(self._encode_jpeg,
                          feed_dict={self._image: image})


def _process_image(filename, coder):
    """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB properly.
    assert len(image.shape) == 3
    assert image.shape[2] == 3

    height = image.shape[0]
    width = image.shape[1]

    image_buffer = coder.encode_jpeg(image)

    return image_buffer, height, width
