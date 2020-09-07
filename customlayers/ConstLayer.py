from tensorflow.keras.layers import Layer
import tensorflow as tf
import training.config as config


class ConstLayer(Layer):
    def __init__(self, shape, *args, **kwargs):
        super(ConstLayer, self).__init__(*args, **kwargs)

        self.shape = shape
        self.constant_start = tf.Variable(tf.ones(self.shape),
                                          name='const_start',
                                          dtype=tf.float32,
                                          trainable=False)
        self.output_dim = [
            config.batch_size, self.shape[0], self.shape[1], self.shape[2]
        ]

    def call(self, inputs=None):
        return self.constant_start

    def compute_output_shape(self):
        return self.output_dim