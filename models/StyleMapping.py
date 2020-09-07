from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Lambda, BatchNormalization
import numpy as np
import training.config as config
import tensorflow as tf
import customlayers.ops as ops
from customlayers.DenseScaled import DenseScaled
from customlayers.StyleMixer import StyleMixer


class StyleMapping(Model):
    def __init__(self,
                 resolution,
                 latent_size=512,
                 layers=8,
                 mixing_prop=0.9,
                 w_ema_decay=0.99,
                 truncation_psi=0.7):
        super(StyleMapping, self).__init__()
        self.resolution = resolution
        self.log2resolution = int(np.math.log2(self.resolution))
        self.latent_size = latent_size
        self.lay = []
        self.style_mixer = StyleMixer(resolution=resolution,
                                      latent_size=latent_size,
                                      mixing_prop=mixing_prop,
                                      w_ema_decay=w_ema_decay,
                                      truncation_psi=truncation_psi)
        
        for i in range(layers):
            self.lay.append(
                DenseScaled(latent_size, lrmul=0.01, gain=np.sqrt(2)))
            self.lay.append(LeakyReLU(config.leaky_relu))

    def call(self, style1, style2):
        style1 = tf.cast(style1, dtype=tf.float32)
        style2 = tf.cast(style2, dtype=tf.float32)

        x = ops.pixel_norm(style1)
        second_style_for_mixing = ops.pixel_norm(style2)
        for lay in self.lay:
            x = lay(x)
            second_style_for_mixing = lay(second_style_for_mixing)

        x = self.style_mixer(x, second_style_for_mixing)
        return x

    def truncation_call(self, style1, style2, truncation):
        style1 = tf.cast(style1, dtype=tf.float32)
        style2 = tf.cast(style2, dtype=tf.float32)

        x = ops.pixel_norm(style1)
        second_style_for_mixing = ops.pixel_norm(style2)
        for lay in self.lay:
            x = lay(x)
            second_style_for_mixing = lay(second_style_for_mixing)
        x = self.style_mixer.w_avg + truncation * (x - self.style_mixer.w_avg)
        second_style_for_mixing = self.style_mixer.w_avg + truncation * (
            second_style_for_mixing - self.style_mixer.w_avg)
        x = self.style_mixer(x, second_style_for_mixing)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.log2resolution * 2 - 2, self.latent_size)
