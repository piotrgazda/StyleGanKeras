from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, LeakyReLU, Reshape, Input, Lambda
from customlayers.EpilogueLayer import EpilogueLayer
import numpy as np
import tensorflow as tf
from customlayers.ProGanGeneratorBlock import ProGanGeneratorBlock
from PIL import Image
import training.config as conf
from matplotlib import cm
from customlayers.CombiningLayer import CombiningLayer
import customlayers.ops as ops
from customlayers.ConvScaled import ConvScaled
from customlayers.DenseScaled import DenseScaled
from customlayers.PixelNormalization import PixelNormalization
import utilities.directories as directories


class ProGanGenerator(Model):
    def __init__(self,
                 resolution=4,
                 latent_size=512,
                 use_noise=True,
                 use_instance=True,
                 use_styles=True,
                 use_pixel=False,
                 *args,
                 **kwargs):
        super(ProGanGenerator, self).__init__(*args, **kwargs)
        self.latent_size = latent_size
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.blocks = []
        for idx in range(3, self.resolution_log2 + 1):
            self.blocks.append(
                ProGanGeneratorBlock(idx, name='generatorblock' + str(idx)))

        self.combining = CombiningLayer(name='combining-gen' +
                                        str(self.resolution_log2))
        self.pixel_normalization1 = PixelNormalization()
        self.pixel_normalization2 = PixelNormalization()
        self.pixel_normalization3 = PixelNormalization()

        self.dense1 = DenseScaled(ops.nf(1) * 16,
                                  gain=np.sqrt(2) / 4,
                                  lrmul=1.0)
        self.leaky_relu1 = LeakyReLU(conf.leaky_relu)

        self.conv1 = ConvScaled(filters=ops.nf(1), kernel_size=3, lrmul=1.0)
        self.leaky_relu2 = LeakyReLU(conf.leaky_relu)

        self.torgb_bigger_res = ConvScaled(filters=3,
                                           kernel_size=1,
                                           lrmul=1.0,
                                           gain=1.0,
                                           name='torgb' +
                                           str(self.resolution_log2))
        self.torgb_lower_res = ConvScaled(filters=3,
                                          kernel_size=1,
                                          lrmul=1.0,
                                          gain=1.0,
                                          name='torgb' +
                                          str(self.resolution_log2 - 1))

        self.output_graph_operation = None
        self.random = tf.random.normal(shape=(conf.batch_size,
                                              self.latent_size))

        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False,
                                       name='global_step' +
                                       str(self.resolution_log2))
        self.increment = tf.assign_add(self.global_step, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3, self.resolution, self.resolution)

    def get_step(self):
        return self.global_step

    def increment_step(self):
        return self.increment

    def call(self, x):
        x = self.pixel_normalization1(x)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.pixel_normalization2(x)
        x = tf.reshape(x, [-1, ops.nf(1), 4, 4])
        x = self.conv1(x)
        x = self.leaky_relu2(x)
        x = self.pixel_normalization3(x)

        for idx in range(len(self.blocks) - 1):
            x = self.blocks[idx](x)

        before = x
        x = self.blocks[-1](x)
        x = self.torgb_bigger_res(x)
        before = self.torgb_lower_res(before)
        before = ops.upscale2d(before)
        x = self.combining([before, x])
        return x

    def truncation_call(self, truncation):
        return self.call(truncation * self.random)

    def return_assign_alpha_weight_operation(self):
        return self.combining.return_assign_alpha_weight_operation()

    def call_without_input(self):
        if self.output_graph_operation is None:
            self.output_graph_operation = self.call(self.random)
        return self.output_graph_operation

    def get_mapper_name_with_path(self, epoch=0):
        return 'mappers/' + 'progan1-e{}-r{}.h5'.format(epoch, self.resolution)

    def get_generator_name_with_path(self, epoch=0):
        return 'generators/' + 'progan1-e{}-r{}.h5'.format(
            epoch, self.resolution)

    def save_mapper(self, epoch=0, name=None):
        return None

    def save_generator(self, epoch=0, name=None):
        if name is None:
            name = directories.get_model_path(
            ) + self.get_generator_name_with_path(epoch=epoch)
        self.save_weights(name)
        return 'progan1-e{}-r{}.h5'.format(epoch, self.resolution)

    def load_mapper(self, mapper_filename):
        return None

    def load_generator(self, generator_filename):
        self.load_weights(generator_filename, by_name=True)
