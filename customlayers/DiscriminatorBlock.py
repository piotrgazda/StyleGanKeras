from tensorflow.keras.layers import Conv2D, LeakyReLU, Layer
from tensorflow.keras.models import Model
import numpy as np
from customlayers.ClipWeights import ClipWeights
from training import config
import customlayers.ops as ops
from customlayers.ConvScaled import ConvScaled
from customlayers.DownscaleConv import DownscaleConv


class DiscriminatorBlock(Model):
    def __init__(self, res, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.res = res
        self.conv1 = ConvScaled(filters=ops.nf(res - 1),
                                kernel_size=3,
                                lrmul=1.0,
                                gain=np.sqrt(2))
        self.relu1 = LeakyReLU(config.leaky_relu)
        self.downscale_conv = DownscaleConv(filters=ops.nf(res - 2),
                                            kernel_size=3,
                                            lrmul=1.0)

        self.relu2 = LeakyReLU(config.leaky_relu)

    def build(self, input_shape):
        self.outputdim = (input_shape[0], ops.nf(self.res - 2),
                          2**(self.res - 1), 2**(self.res - 1))

    def compute_output_shape(self, input_shape):
        return self.outputdim

    def call(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = ops.blur2d(y)
        y = self.downscale_conv(y)
        y = self.relu2(y)
        return y
