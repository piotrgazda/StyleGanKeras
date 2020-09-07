from tensorflow.keras.layers import Layer, Flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class DenseScaled(Layer):
    def __init__(self, units, lrmul=1.0, gain=np.sqrt(2), *args, **kwargs):
        super(DenseScaled, self).__init__(*args, **kwargs)
        self.gain = gain
        self.lrmul = lrmul
        self.units = units
        self.lrmul_tensor = tf.constant(lrmul, dtype=tf.float32)

    def build(self, input_shape):
        super(DenseScaled, self).build(input_shape)
        units_in_input = input_shape[-1].value
        weight_shape = (units_in_input, self.units)
        fan_in = units_in_input
        he_std = self.gain / np.sqrt(fan_in)
        init_std = 1.0 / self.lrmul
        self.runtime_coef = tf.constant(he_std * self.lrmul, dtype=tf.float32)

        self.dense_weights = self.add_weight(
            name='dense_weight',
            shape=weight_shape,
            initializer=tf.initializers.random_normal(0.0, init_std),
            trainable=True,
            dtype=tf.float32)

        self.dense_biases = self.add_weight(
            name='dense_bias',
            shape=self.units,
            initializer=tf.initializers.zeros(),
            dtype=tf.float32,
            trainable=True)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.units)
        return output_shape

    def call(self, x):
        y = tf.matmul(
            x, self.dense_weights *
            self.runtime_coef) + self.dense_biases * self.lrmul_tensor
        return y
