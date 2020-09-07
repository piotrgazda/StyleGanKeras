from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, LeakyReLU
from customlayers.ConvScaled import ConvScaled
from customlayers.DownscaleConv import DownscaleConv
from customlayers.PixelNormalization import PixelNormalization
import training.config as config
import customlayers.ops as ops


class ProGanDiscriminatorBlock(Model):
    def __init__(self, res, *args, **kwargs):
        super(ProGanDiscriminatorBlock, self).__init__(*args, **kwargs)
        self.resolution = res
        self.conv = ConvScaled(filters=ops.nf(res - 1), kernel_size=3, lrmul=1.0)
        self.downscale_conv = DownscaleConv(filters=ops.nf(res - 2),
                                            kernel_size=3,
                                            lrmul=1.0)
        self.leaky_relu1 = LeakyReLU(config.leaky_relu)
        self.leaky_relu2 = LeakyReLU(config.leaky_relu)

    def call(self, x):
        x = self.conv(x)
        x = self.leaky_relu1(x)
        x = self.downscale_conv(x)
        x = self.leaky_relu2(x)
        return x
