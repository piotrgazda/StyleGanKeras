import sys
import tensorflow as tf
import numpy as np
from training.Trainer import Trainer
import training.config as config
import utilities.builders as builders
import utilities.directories as directories
import utilities.utils as utils
from utilities.ArgumentParser import ArgumentParser
import metrics.inception_processing as inception_processing
tf.logging.set_verbosity(tf.logging.ERROR)

parser = ArgumentParser(sys.argv)
resolution = parser.get_resolution()
starting_epoch = parser.get_starting_epoch()
dataset_destination = config.data_unformatted_name.format(
    parser.get_resolution_log2())
utils.mount_google_disc(parser.is_google_colab())
if parser.is_progan():
    names = parser.get_default_load_model_names('progan1-e{}-r{}.h5')
else:
    names = parser.get_default_load_model_names('stylegan1-e{}-r{}.h5')

metrics_list = utils.get_metrics_list(parser.use_metrics())
loss = utils.get_loss()
while True:
    utils.copy_dataset(dataset_destination, resolution,
                       parser.is_google_colab())

    sess = tf.Session()
    with sess.as_default():
        if parser.is_progan():
            generator = builders.build_progan_generator(resolution=resolution)
            discriminator = builders.build_progan_discriminator(
                resolution=resolution)
        else:
            mapper = builders.build_mapper(resolution=resolution)
            generator = builders.build_generator(resolution=resolution)
            generator = builders.synthesize_models(mapper, generator)
            discriminator = builders.build_discriminator(resolution=resolution)

        trainer = Trainer(tf_session=sess,
                          generator=generator,
                          discriminator=discriminator)
        sess.run(tf.global_variables_initializer())

        names = trainer.train_epochs(
            loss=loss,
            starting_epoch=starting_epoch,
            epochs=config.epochs,
            iterations_per_epoch=config.iterations_per_epoch,
            metrics_list=metrics_list,
            names=names)
    resolution *= 2
    parser.arguments.resolution *= 2
    dataset_destination = config.data_unformatted_name.format(
        parser.get_resolution_log2())
    starting_epoch = 0
    inception_processing.inception_model = None
    tf.keras.backend.clear_session()
