from tensorflow.keras.layers import Layer
import tensorflow as tf


class PixelNormalization(Layer):
    def call(self, x, epsilon=1e-8):
        return x * tf.rsqrt(
            tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
