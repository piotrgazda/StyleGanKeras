from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


class DownscaleConv(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 lrmul=1.0,
                 gain=np.sqrt(2),
                 use_bias=True,
                 *args,
                 **kwargs):
        super(DownscaleConv, self).__init__(*args, **kwargs)
        self.filters_out = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        self.filters_in = input_shape[1].value
        self.resolution = input_shape[2].value

        fan_in = self.filters_in * self.kernel_size * self.kernel_size
        he_std = self.gain / np.sqrt(fan_in)
        init_std = 1.0 / self.lrmul
        self.runtime_coef = he_std * self.lrmul

        weight_shape = (self.kernel_size, self.kernel_size, self.filters_in,
                        self.filters_out)
        self.conv_weights = self.add_weight(
            name='conv-weights',
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0.0, init_std),
            trainable=True,
            dtype=tf.float32)
        if self.use_bias:
            self.conv_biases = self.add_weight(
                name='conv-biases',
                shape=(self.filters_out, self.resolution // 2,
                       self.resolution // 2),
                initializer=tf.initializers.zeros(),
                dtype=tf.float32,
                trainable=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.filters_out, self.resolution // 2,
                self.resolution // 2)

    def call(self, x):
        w = tf.pad(self.conv_weights * self.runtime_coef,
                   [[1, 1], [1, 1], [0, 0], [0, 0]],
                   mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(
            x, w, strides=[1, 1, 2, 2], padding='SAME',
            data_format='NCHW') + self.conv_biases * self.lrmul
