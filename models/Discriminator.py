from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, AveragePooling2D, Input, Flatten, Dropout
from customlayers.DiscriminatorBlock import DiscriminatorBlock
from customlayers.MinibatchStdDev import MinibatchStdDev
import math
import tensorflow as tf
import numpy as np
from customlayers.ClipWeights import ClipWeights
from training import config
import customlayers.ops as ops
from customlayers.DenseScaled import DenseScaled
from customlayers.ConvScaled import ConvScaled
from customlayers.Bias import Bias
from customlayers.CombiningLayer import CombiningLayer
import utilities.directories as directories


class Discriminator(Model):
    def __init__(self, resolution, mbstd_group_size=4, mbstd_num_features=1):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.fromrgb_higher_res = ConvScaled(
            filters=ops.nf(self.resolution_log2 - 1),
            kernel_size=1,
            lrmul=1.0,
            gain=np.sqrt(2),
            name='fromrgb' + str(self.resolution_log2))
        self.fromrgb_lower_res = ConvScaled(
            filters=ops.nf(self.resolution_log2 - 2),
            kernel_size=1,
            lrmul=1.0,
            gain=np.sqrt(2),
            name='fromrgb' + str(self.resolution_log2 - 1))
        self.combining = CombiningLayer(name='combining' +
                                        str(self.resolution_log2))

        self.blocks = []
        for idx in range(self.resolution_log2, 2, -1):
            self.blocks.append(
                DiscriminatorBlock(idx, name='discriminator_block' + str(idx)))

        self.relu1 = LeakyReLU(config.leaky_relu)
        self.minibatchdev = MinibatchStdDev(group_size=4, num_new_features=1)
        self.conv1 = ConvScaled(filters=ops.nf(1),
                                kernel_size=3,
                                lrmul=1.0,
                                gain=np.sqrt(2))
        self.relu2 = LeakyReLU(config.leaky_relu)
        self.flatten = Flatten()
        self.dense1 = DenseScaled(units=ops.nf(0), lrmul=1.0, gain=np.sqrt(2))
        self.relu3 = LeakyReLU(config.leaky_relu)
        self.dense2 = DenseScaled(units=1, lrmul=1.0, gain=1.0)
        self.relu_lower_res = LeakyReLU(config.leaky_relu)

    def call(self, inputs):
        beforeblock = inputs
        y = self.fromrgb_higher_res(inputs)
        y = self.relu1(y)
        y = self.blocks[0](y)
        missing_block = ops.downscale2d(beforeblock)
        missing_block = self.fromrgb_lower_res(missing_block)
        missing_block = self.relu_lower_res(missing_block)
        y = self.combining([missing_block, y])
        for idx in range(1, len(self.blocks)):
            y = self.blocks[idx](y)

        y = self.minibatchdev(y)
        y = self.conv1(y)
        y = self.relu2(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.relu3(y)
        y = self.dense2(y)
        return y

    def compute_output_shape(self, input_shape):
        return [input_shape[0], 1]

    def get_model_name_with_path(self, epoch=0):
        return 'discriminators/' + 'stylegan1-e{}-r{}.h5'.format(
            epoch, self.resolution)

    def return_assign_alpha_weight_operation(self):
        return self.combining.return_assign_alpha_weight_operation()

    def save(self, epoch=0, name=None):
        if name is None:
            name = directories.get_model_path(
            ) + self.get_model_name_with_path(epoch=epoch)
        self.save_weights(name)
        return 'stylegan1-e{}-r{}.h5'.format(epoch, self.resolution)

    def load(self, name):
        self.load_weights(name, by_name=True)

    def return_alpha(self):
        return self.combining.return_alpha()