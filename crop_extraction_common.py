
import numpy as np
import tensorflow as tf
import cv2


class ShardedRecordWriter(object):

    def __init__(self, path_format, examples_per_shard):
        self._path_format = path_format
        self._examples_per_shard = examples_per_shard
        self._shard_index = 0
        self._example_index_in_shard = 0
        self._new_file()

    def _new_file(self):
        if self._shard_index > 0:
            self._writer.close()
        self._writer = tf.python_io.TFRecordWriter(self._path_format.format(self._shard_index))
        self._shard_index += 1
        self._example_index_in_shard = 0

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.close()

    def write(self, serialised_example):
        if self._example_index_in_shard == self._examples_per_shard:
            self._new_file()
        self._writer.write(serialised_example)
        self._example_index_in_shard += 1

    def close(self):
        self._writer.close()


def float32_feature(value):
    value = np.asarray([value]).flatten()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def jpeg_feature(image):
    value = cv2.imencode('.jpg', image)[1].tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def png_feature(image):
    value = cv2.imencode('.png', image)[1].tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


