from models.StyleGenerator import StyleGenerator
from models.StyleMapping import StyleMapping
import utilities.builders as builders
import training.config as config
import tensorflow as tf
import numpy as np
from data.ImageOutGenerator import ImageOutGenerator
from tensorflow.keras.layers import Input
from PIL import Image
resolution = 128
image_out = ImageOutGenerator()


def initialize_load():
    random = np.random.normal(size=(config.batch_size, config.latent_size))
    mapped = style_mapper(random, random)
    image = generator(mapped)



def get_random_images():
    randomized = tf.random_normal(shape=(config.batch_size,
                                         config.latent_size))
    style = mapper(randomized, randomized)
    image = generator(style)
    return image_out.get_pillow_image(image.numpy(), resolution, cols=4)


def get_style_mix(cols, rows, truncaiton, column_orientation=True):

    images = image_out.generate_style_mix_image_eager(
        mapper,
        generator,
        resolution,
        truncation=truncaiton,
        cols=cols,
        rows=rows,
        column_orientation=column_orientation)

    #images = sess.run(y)

    return images


def get_truncation(rows, truncation_step):
    images = []
    for i in range(rows):
        images.append(
            image_out.generate_truncation_diagram_eager(
                mapper,
                generator,
                resolution,
                truncation_step,
                convert_to_pillow=False))

    return Image.fromarray(np.uint8(np.concatenate(images, axis=0)))


tf.enable_eager_execution()
input_shape = (config.batch_size, config.latent_size)

mapper = builders.build_mapper(resolution)
generator = builders.build_generator(resolution)
get_random_images()
tf.global_variables_initializer()
mapper.load_weights('../models-save/mappers/mapper.h5', by_name=True)
generator.load_weights('../models-save/generators/generator.h5',
                       by_name=True)
