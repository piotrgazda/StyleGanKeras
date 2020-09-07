from tensorflow.keras.layers import Layer, Lambda
import tensorflow as tf
import numpy as np


def lerp(a, b, t):
    out = a + (b - a) * t
    return out


class StyleMixer(Layer):
    def __init__(self,
                 resolution,
                 latent_size=512,
                 mixing_prop=0.9,
                 w_ema_decay=0.99,
                 truncation_psi=0.7):
        super(StyleMixer, self).__init__()
        self.latent_size = latent_size
        self.resolution = resolution
        self.log2resolution = int(np.math.log2(self.resolution))
        self.mixing_prop = mixing_prop
        self.w_ema_decay = w_ema_decay
        self.truncation_psi = truncation_psi

    def build(self, input_shape):
        super(StyleMixer, self).build(input_shape)

        self.w_avg = self.add_weight(name='w_avg',
                                     shape=(self.latent_size, ),
                                     initializer=tf.initializers.zeros(),
                                     trainable=False,
                                     dtype=tf.float32)

    def truncation_trick(self, n_broadcast, w_broadcasted, w_avg,
                         truncation_psi):
        with tf.variable_scope('truncation'):
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_indices.shape, dtype=np.float32)
            coefs = tf.where(layer_indices < self.truncation_cutoff,
                             truncation_psi * ones, ones)
            w_broadcasted = lerp(w_avg, w_broadcasted, coefs)
        return w_broadcasted

 
    def style_mixing_regularization(self, style1, style2, log2resolution):
        layer_indices = np.arange(log2resolution * 2 -
                                  2)[np.newaxis, :, np.newaxis]
        last_layer_index = log2resolution * 2 - 2

        mixing_cutoff = tf.cond(
            tf.less(tf.random_uniform([], 0.0, 1.0), self.mixing_prop),
            lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
            lambda: tf.constant(last_layer_index, dtype=tf.int32))

        style1 = tf.where(
            tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(style1)),
            style1, style2)
        return style1

    def update_moving_average_of_w(self, w_broadcasted, w_avg):
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
        update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, self.w_ema_decay))

        with tf.control_dependencies([update_op]):
            w_broadcasted = tf.identity(w_broadcasted)
        return w_broadcasted

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.log2resolution, self.latent_size)

    def if_training(self, style1, style2):
        style1 = self.update_moving_average_of_w(style1, self.w_avg)
        style1 = self.style_mixing_regularization(
            style1=style1, style2=style2, log2resolution=self.log2resolution)
        return style1

    def call(self, style1, style2):

        style1 = tf.reshape(style1, shape=(-1, 1, 512))
        style1 = tf.tile(style1, multiples=[1, self.log2resolution * 2 - 2, 1])
        style2 = tf.reshape(style2, shape=(-1, 1, 512))
        style2 = tf.tile(style2, multiples=[1, self.log2resolution * 2 - 2, 1])
        style1 = self.if_training(style1, style2)
        #else:
        #    w_broadcasted = self.truncation_trick(n_broadcast, w_broadcasted,
        #                                          w_avg, self.truncation_psi)

        return style1
