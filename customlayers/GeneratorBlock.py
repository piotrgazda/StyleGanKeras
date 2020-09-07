from tensorflow.keras.layers import Conv2D, LeakyReLU, Layer
import numpy as np
from tensorflow.keras.models import Model
import customlayers.ops as ops
from customlayers.ConvScaled import ConvScaled
from customlayers.EpilogueLayer import EpilogueLayer
import training.config as config
from customlayers.CombiningLayer import CombiningLayer
from customlayers.UpscaleConv import UpscaleConv


class GeneratorBlock(Model):
    def __init__(self, res, **kwargs):
        super(GeneratorBlock, self).__init__(**kwargs)
        self.res = res
        self.upscale_conv = UpscaleConv(filters=ops.nf(res - 1),
                                        kernel_size=3,
                                        lrmul=1.0)
        self.conv1 = (ConvScaled(filters=ops.nf(res - 1),
                                 kernel_size=3,
                                 lrmul=1.0,
                                 gain=np.sqrt(2)))
        self.epilogue1 = EpilogueLayer(dlatents_size=config.latent_size)
        self.epilogue2 = EpilogueLayer(dlatents_size=config.latent_size)

    def build(self, input_shape):
        self.outputdim = (input_shape[0], ops.nf(self.res - 1), 2**(self.res),
                          2**(self.res))

    def compute_output_shape(self, input_shape):
        return self.outputdim

    def call(self, input):
        x = input[0]
        styles = input[1]
        y = self.upscale_conv(x)
        y = ops.blur2d(y)
        y = self.epilogue1([y, styles[:, 0]])
        y = self.conv1(y)
        y = self.epilogue2([y, styles[:, 1]])
        return y

