from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, LeakyReLU
from customlayers.PixelNormalization import PixelNormalization
from customlayers.UpscaleConv import UpscaleConv
from customlayers.ConvScaled import ConvScaled
import customlayers.ops as ops
import training.config as config


class ProGanGeneratorBlock(Model):
    def __init__(self, res, *args, **kwargs):
        super(ProGanGeneratorBlock, self).__init__(*args, **kwargs)
        self.log2res = res
        self.pixel_normaliation1 = PixelNormalization()
        self.pixel_normaliation2 = PixelNormalization()
        self.upscale_conv = UpscaleConv(filters=ops.nf(res - 1),
                                        kernel_size=3,
                                        lrmul=1.0)
        self.conv_scaled = ConvScaled(filters=ops.nf(res - 1),
                                      kernel_size=3,
                                      lrmul=1.0)
        self.leaky_relu1 = LeakyReLU(config.leaky_relu)
        self.leaky_relu2 = LeakyReLU(config.leaky_relu)

    def build(self, input_shape):
        self.outputdim = (input_shape[0], ops.nf(self.log2res - 1),
                          2**(self.log2res + 1), 2**(self.log2res + 1))

    def compute_output_shape(self, input_shape):
        return self.outputdim

    def call(self, x):
        x = self.upscale_conv(x)
        x = self.leaky_relu1(x)
        x = self.pixel_normaliation1(x)
        x = self.conv_scaled(x)
        x = self.leaky_relu2(x)
        x = self.pixel_normaliation2(x)
        return x