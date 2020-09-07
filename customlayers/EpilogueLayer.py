from tensorflow.keras.layers import Layer, LeakyReLU
import tensorflow as tf
import customlayers.ops as ops
import training.config as config
import numpy as np


class EpilogueLayer(Layer):
    def __init__(self,
                 dlatents_size=None,
                 use_noise=True,
                 use_pixel_norm=False,
                 use_instance_norm=True,
                 randomize_noise=True,
                 use_styles=True,
                 lrmul=1.0,
                 gain=1.0,
                 **kwargs):
        super(EpilogueLayer, self).__init__(**kwargs)
        self.use_noise = use_noise
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.randomize_noise = randomize_noise
        self.use_styles = use_styles
        self.dlatents_size = dlatents_size
        self.leakyrelu = LeakyReLU(config.leaky_relu)
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        super(EpilogueLayer, self).build(input_shape)
        self.output_dim = input_shape[0]
        self.noisebias = self.add_weight(name='noisebias',
                                         shape=(self.output_dim[1].value,
                                                self.output_dim[2].value,
                                                self.output_dim[3].value),
                                         initializer=tf.initializers.zeros(),
                                         trainable=True,
                                         dtype=tf.float32)

        self.stylebias = self.add_weight(name='stylebias',
                                         shape=(self.output_dim[1].value *
                                                2, ),
                                         initializer=tf.initializers.zeros(),
                                         trainable=True,
                                         dtype=tf.float32)
        self.noiseweight = self.add_weight(name='noiseweight',
                                           shape=(self.output_dim[1].value, ),
                                           initializer=tf.initializers.zeros(),
                                           dtype=tf.float32,
                                           trainable=True)
        if self.dlatents_size is not None:
            fan_in = self.dlatents_size
            he_std = self.gain / np.sqrt(fan_in)  # He init

            # equalized learning rate
            init_std = 1.0 / self.lrmul
            self.runtime_coef = he_std * self.lrmul
            self.styleweight = self.add_weight(
                'styleweights',
                shape=(self.dlatents_size, self.output_dim[1].value * 2),
                trainable=True,
                initializer=tf.initializers.random_normal(0.0, init_std))

    def call(self, inputs, noise_inputs=None):
        x = inputs[0]
        dlatents_in = inputs[1]
        if self.use_noise:
            x = ops.apply_noise(x,
                                weight=self.noiseweight,
                                noise_var=noise_inputs,
                                randomize_noise=self.randomize_noise)
        x = x + self.noisebias
        x = self.leakyrelu(x)
        if self.use_pixel_norm:
            x = ops.pixel_norm(x)
        if self.use_instance_norm:
            x = ops.instance_norm(x)
        if self.use_styles:
            x = ops.style_mod(x,
                              dlatents_in,
                              weights=self.styleweight * self.runtime_coef,
                              bias=self.stylebias * self.lrmul)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]
