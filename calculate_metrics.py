import sys
import tensorflow as tf
from metrics.FrechettInceptionDistance import FrechettInceptionDistance
from metrics.InceptionScore import InceptionScore
from metrics.GeometryScore import GeometryScore
from metrics.DiscriminativeMetric import DiscriminativeMetric

import utilities.builders as builders
import utilities.utils as utils
import training.config as config
import utilities.directories as directories
from utilities.ArgumentParser import ArgumentParser
from data.DataReader import DataReader
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

parser = ArgumentParser(sys.argv)
epoch = parser.get_starting_epoch()
resolution = parser.get_resolution()

utils.mount_google_disc(parser.is_google_colab())
dataset_destination = config.data_unformatted_name.format(
    parser.get_resolution_log2())

utils.copy_dataset(dataset_destination, resolution, parser.is_google_colab())

if parser.is_progan():
    generator = builders.build_progan_generator(resolution)
    discriminator = builders.build_progan_discriminator(resolution)

else:
    mapper = builders.build_mapper(resolution)
    generator = builders.build_generator(resolution)
    discriminator = builders.build_discriminator(resolution)
    generator = builders.synthesize_models(mapper, generator)

log2res = int(np.math.log2(resolution))
unformatted_filename = config.data_unformatted_name
dataset_name = unformatted_filename.format(log2res)
reader = DataReader(dataset_name, resolution, shuffle_mb=1024)

metrics_list = [
    DiscriminativeMetric(),
    FrechettInceptionDistance(),
    InceptionScore()
]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if parser.is_progan():
        names = parser.get_default_load_model_names('progan1-e{}-r{}.h5')
        discriminator.load(directories.get_discriminators_folder() + names[0])
        generator.load_generator(directories.get_generators_folder() +
                                 names[1])
    else:
        names = parser.get_default_load_model_names('stylegan1-e{}-r{}.h5')
        generator.load_mapper(directories.get_mappers_folder() + names[0])
        discriminator.load(directories.get_discriminators_folder() + names[0])
        generator.load_generator(directories.get_generators_folder() +
                                 names[1])

    for metric in metrics_list:
        metric.evaluate(sess, generator, reader.get_batch())
