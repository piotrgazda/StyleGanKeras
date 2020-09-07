import tensorflow as tf
import numpy as np
from training import config
import time


class DataReader:
    def __init__(self,
                 dataset_filename,
                 resolution,
                 shuffle_mb=1024,
                 buffer_mb=256):
        self.dataset_filename = dataset_filename
        self.resolution = resolution
        self.data_set = tf.data.TFRecordDataset(self.dataset_filename,
                                                buffer_size=buffer_mb << 10)
        image_bytes = self.resolution * self.resolution * 3
        to_shuffle = (shuffle_mb << 10) // image_bytes
        self.data = self.data_set.shuffle(to_shuffle).repeat().map(
            parse_tfrecord_tf).batch(config.batch_size).apply(
                tf.data.experimental.ignore_errors())
        self.iteration = self.data.make_one_shot_iterator().get_next()

    def get_batch(self):
        return self.iteration


def parse_tfrecord_tf(dataset):
    features = tf.io.parse_single_example(dataset,
                                          features={
                                              'shape':
                                              tf.io.FixedLenFeature([3],
                                                                    tf.int64),
                                              'data':
                                              tf.io.FixedLenFeature([],
                                                                    tf.string)
                                          })
    data = tf.io.decode_raw(features['data'], tf.uint8)
    data = tf.reshape(data, features['shape'])
    data = tf.cond(
        tf.random.uniform([], 0.0, 1.0) > 0.5,
        lambda: tf.reverse(data, axis=[-1]), lambda: data)
    return tf.math.scalar_mul(scalar=1 / 127.5,
                              x=tf.cast(data, dtype=tf.float32)) - 1
