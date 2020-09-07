from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class ConvScaled(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 lrmul=1.0,
                 gain=np.sqrt(2),
                 use_bias=True,
                 *args,
                 **kwargs):
        super(ConvScaled, self).__init__(*args, **kwargs)
        self.filters_out = filters
        self.kernel_size = kernel_size
        self.lrmul = lrmul
        self.gain = gain
        self.use_bias = use_bias
        self.name_for_bias = 'conv-biases'

    def build(self, input_shape):
        super(ConvScaled, self).build(input_shape)
        self.filters_in = input_shape[1].value
        self.resolution = input_shape[2].value
        fan_in = self.filters_in * self.kernel_size * self.kernel_size
        he_std = self.gain / np.sqrt(fan_in)  # He init

        # equalized learning rate
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
                name=self.name_for_bias,
                shape=(self.filters_out, self.resolution, self.resolution),
                initializer=tf.initializers.zeros(),
                dtype=tf.float32,
                trainable=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.filters_out, self.resolution,
                self.resolution)

    def call(self, x):
        x = tf.nn.conv2d(input=x,
                         filter=self.conv_weights * self.runtime_coef,
                         padding='SAME',
                         strides=[1, 1, 1, 1],
                         data_format='NCHW')
        if self.use_bias:
            x = x + self.conv_biases * self.lrmul
        return x
