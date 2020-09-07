from tensorflow.keras.layers import Layer
import tensorflow as tf


class Bias(Layer):
    def __init__(self, lrmul=1.0, *args, **kwargs):
        super(Bias, self).__init__(*args, **kwargs)
        self.lrmul = lrmul

    def build(self, input_shape):
        super(Bias, self).build(input_shape)
        self.bias = self.add_weight(name='bias-weight',
                                    shape=(input_shape[1], input_shape[2],
                                           input_shape[3]),
                                    initializer=tf.initializers.zeros(),
                                    dtype=tf.float32)

    def call(self, x):
        return x + self.bias * self.lrmul
