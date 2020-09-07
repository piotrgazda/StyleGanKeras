from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.StyleMapping import StyleMapping
from models.Discriminator import Discriminator
from models.StyleGenerator import StyleGenerator
from models.StyleGanSynthesizedGenerator import StyleGanSynthesizedGenerator
from models.ProGanGenerator import ProGanGenerator
from models.ProGanDiscriminator import ProGanDiscriminator

import training.config as config
import tensorflow as tf
import numpy as np


def build_discriminator(resolution, discriminator_save=None):
    discriminator = Discriminator(resolution=resolution)

    discriminator_input_shape = (3, resolution, resolution)
    discriminator_input_layer = Input(shape=discriminator_input_shape)
    discriminator_output = discriminator(discriminator_input_layer)
    return discriminator


def build_mapper(resolution,
                 layers=8,
                 latent_size=config.latent_size,
                 mapper_save=None):
    log2res = int(np.math.log2(resolution))
    mapper = StyleMapping(latent_size=config.latent_size,
                          layers=layers,
                          resolution=resolution)
    mapper_input_shape = (config.latent_size, )
    mapper_input_layer1 = Input(shape=mapper_input_shape)
    mapper_input_layer2 = Input(shape=mapper_input_shape)

    mapper_output = mapper(mapper_input_layer1, mapper_input_layer2)
    return mapper


def build_generator(resolution, latent_size=config.latent_size):
    log2res = int(np.math.log2(resolution))
    generator = StyleGenerator(resolution=resolution,
                               latent_size=config.latent_size)
    generator_input_shape = (log2res * 2 - 2, latent_size)
    generator_input_layer = Input(shape=generator_input_shape)
    generator_output = generator(generator_input_layer)
    return generator


def synthesize_models(mapper, generator, name='stylegan1'):
    return StyleGanSynthesizedGenerator(mapper, generator, name)


def load_model(model, file):
    model.load_weights(file, by_name=True)


def build_progan_generator(resolution):
    generator = ProGanGenerator(resolution)
    generator_input_shape = (512, )
    generator_input_layer = Input(shape=generator_input_shape)
    generator_output = generator(generator_input_layer)
    return generator


def build_progan_discriminator(resolution):
    discriminator = ProGanDiscriminator(resolution)
    discriminator_input_shape = (3, resolution, resolution)
    discriminator_input_layer = Input(discriminator_input_shape)
    discriminator_output = discriminator(discriminator_input_layer)
    return discriminator
