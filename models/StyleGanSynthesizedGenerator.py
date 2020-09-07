from tensorflow.keras.models import Model
import tensorflow as tf
import training.config as config
import utilities.directories as directories


class StyleGanSynthesizedGenerator(Model):
    def __init__(self, mapper, generator, name, batch_size=config.batch_size):
        super(StyleGanSynthesizedGenerator, self).__init__()
        self.mapper = mapper
        self.generator = generator
        self.resolution = self.generator.resolution
        self.name_model = name
        self.random1 = tf.random.normal(shape=(batch_size,
                                               self.mapper.latent_size))
        self.random2 = tf.random.normal(shape=(batch_size,
                                               self.mapper.latent_size))
        self.output_graph_operation = None

        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False,
                                       name='global_step' +
                                       str(self.resolution))

        self.increment = tf.assign_add(self.global_step, 1)

    def call_without_input(self):
        if self.output_graph_operation is None:
            self.output_graph_operation = self.call(self.random1, self.random2)
        return self.output_graph_operation

    def call(self, random1, random2):
        y = self.mapper(random1, random2)
        y = self.generator(y)
        return y

    def truncation_call(self, truncation):
        y = self.mapper.truncation_call(self.random1, self.random2, truncation)
        y = self.generator(y)
        return y

    def get_mapper_name_with_path(self, epoch=0):
        return 'mappers/' + self.name_model + '-e{}-r{}.h5'.format(
            epoch, self.resolution)

    def get_generator_name_with_path(self, epoch=0):
        return 'generators/' + self.name_model + '-e{}-r{}.h5'.format(
            epoch, self.resolution)

    def return_assign_alpha_weight_operation(self):
        return self.generator.return_assign_alpha_weight_operation()

    def save_mapper(self, epoch=0, name=None):
        if name is None:
            name = directories.get_model_path(
            ) + self.get_mapper_name_with_path(epoch=epoch)
        self.mapper.save_weights(name)
        return self.name_model + '-e{}-r{}.h5'.format(epoch, self.resolution)

    def save_generator(self, epoch=0, name=None):
        if name is None:
            name = directories.get_model_path(
            ) + self.get_generator_name_with_path(epoch=epoch)
        self.generator.save_weights(name)
        return self.name_model + '-e{}-r{}.h5'.format(epoch, self.resolution)

    def load_mapper(self, mapper_filename):
        self.mapper.load_weights(mapper_filename, by_name=True)

    def load_generator(self, generator_filename):
        self.generator.load_weights(generator_filename, by_name=True)

    def get_step(self):
        return self.global_step

    def increment_step(self):
        return self.increment