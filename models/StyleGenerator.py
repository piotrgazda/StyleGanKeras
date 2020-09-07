from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, LeakyReLU, Reshape, Input, Lambda
from customlayers.EpilogueLayer import EpilogueLayer
import numpy as np
import tensorflow as tf
from customlayers.GeneratorBlock import GeneratorBlock
from PIL import Image
import training.config as conf
from matplotlib import cm
from customlayers.CombiningLayer import CombiningLayer
import customlayers.ops as ops
from customlayers.ConvScaled import ConvScaled
from customlayers.StyleMixer import StyleMixer
from customlayers.Bias import Bias
from customlayers.ConstLayer import ConstLayer


class StyleGenerator(Model):

    constant_start_shape = (1, 512, 4, 4)

    def __init__(self,
                 resolution=4,
                 latent_size=512,
                 use_noise=True,
                 use_instance=True,
                 use_styles=True,
                 use_pixel=False):
        super(StyleGenerator, self).__init__()

        self.constant_start = tf.Variable(tf.ones(self.constant_start_shape),
                                          name='const_start',
                                          dtype=tf.float32,
                                          trainable=False)
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.layer_names = []
        
        self.early_block_epilogue1 = EpilogueLayer(
            dlatents_size=latent_size,
            use_noise=use_noise,
            use_instance_norm=use_instance,
            use_styles=use_styles,
            use_pixel_norm=use_pixel,
            lrmul=1.0,
            gain=1.0)

        self.early_block_conv1 = ConvScaled(filters=ops.nf(1),
                                            kernel_size=3,
                                            lrmul=1.0,
                                            gain=np.sqrt(2))

        self.early_block_epilogue2 = EpilogueLayer(
            dlatents_size=latent_size,
            use_noise=use_noise,
            use_instance_norm=use_instance,
            use_pixel_norm=use_pixel,
            use_styles=use_styles,
            lrmul=1.0,
            gain=1.0)
        self.blocks = []

        for idx in range(3, self.resolution_log2 + 1):
            self.blocks.append(
                GeneratorBlock(idx, name='generatorblock' + str(idx)))
        self.torgb_lower_res = ConvScaled(filters=3,
                                          kernel_size=1,
                                          lrmul=1.0,
                                          gain=1.0,
                                          name='torgb' +
                                          str(self.resolution_log2 - 1))
        self.torgb_higher_res = ConvScaled(filters=3,
                                           kernel_size=1,
                                           lrmul=1.0,
                                           gain=1.0,
                                           name='torgb' +
                                           str(self.resolution_log2))

        self.combining = CombiningLayer(name='combining-gen' +
                                        str(self.resolution_log2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3, self.resolution, self.resolution)

    def call(self, styles):
        y = self.early_block_epilogue1([self.constant_start, styles[:, 0, :]])
        y = self.early_block_conv1(y)
        y = self.early_block_epilogue2([y, styles[:, 1, :]])
        styles_idx_to_send = 2
        for idx in range(len(self.blocks) - 1):
            y = self.blocks[idx](
                [y, styles[:, styles_idx_to_send:styles_idx_to_send + 2, :]])
            styles_idx_to_send += 2

        before = self.torgb_lower_res(y)
        before = ops.upscale2d(before)
        y = self.blocks[-1](
            [y, styles[:, styles_idx_to_send:styles_idx_to_send + 2, :]])
        y = self.torgb_higher_res(y)
        y = self.combining([before, y])
        return y

    def return_assign_alpha_weight_operation(self):
        return self.combining.return_assign_alpha_weight_operation()
