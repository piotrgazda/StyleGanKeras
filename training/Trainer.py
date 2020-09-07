import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm

from training import config
from models.Discriminator import Discriminator
from models.StyleGenerator import StyleGenerator
from models.StyleMapping import StyleMapping
from data.ImageOutGenerator import ImageOutGenerator
from data.DataReader import DataReader
import utilities.directories as directories
import training.saver as saver
import utilities.operations as operations
import utilities.builders as builders


class Trainer:
    def __init__(self, tf_session, generator, discriminator):
        self.session = tf_session
        self.generator = generator
        self.discriminator = discriminator
        self.resolution = discriminator.resolution
        self.log2res = int(np.math.log2(self.resolution))

        unformatted_filename = config.data_unformatted_name
        dataset_name = unformatted_filename.format(self.log2res)
        self.initialize_dataset(dataset_name=dataset_name)

        self.image_generator = ImageOutGenerator(
            image_saving_path=directories.images_save_folder)
        self.initialized = False
        self.set_optimizers()
        self.initialize_accumulation_variables()
        self.initialize_zero_operations()
        self.writer = tf.summary.FileWriter(
            directories.get_logs_full_filename(self.resolution),
            self.session.graph)

    def initialize_accumulation_variables(self):

        self.accum_vars_dis = [
            tf.Variable(initial_value=tf.zeros_like(tv),
                        trainable=False,
                        dtype=tf.float32)
            for tv in self.discriminator.trainable_weights
        ]
        self.accum_vars_gen = [
            tf.Variable(initial_value=tf.zeros_like(tv),
                        trainable=False,
                        dtype=tf.float32)
            for tv in self.generator.trainable_weights
        ]

    def initialize_zero_operations(self):
        self.zero_ops_dis = [
            tf.assign(tv, tf.zeros_like(tv)) for tv in self.accum_vars_dis
        ]
        self.zero_ops_gen = [
            tf.assign(tv, tf.zeros_like(tv)) for tv in self.accum_vars_gen
        ]

    def set_optimizers(self):
        self.optimizer_discriminator = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2)
        self.optimized_generator = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2)

    def initialize_models(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def get_data_from_dataset(self):
        from_data_set = self.data_generator.get_batch()
        self.from_data_set = tf.reshape(from_data_set,
                                        shape=(config.batch_size, 3,
                                               self.resolution,
                                               self.resolution))

    def initialize_train_operations(self, loss):

        const = tf.constant(1 / config.minibatches_per_apply)

        tv_dis = self.discriminator.trainable_weights
        tv_gen = self.generator.trainable_weights
        self.get_data_from_dataset()
        real_features = self.from_data_set

        real_features = operations.smooth_crossfade(
            real_features, self.discriminator.return_alpha())

        loss_g, loss_d = loss.evaluate(self.generator, self.discriminator,
                                       real_features)
        gradient_dis = self.optimizer_discriminator.compute_gradients(
            loss_d, tv_dis)
        gradient_gen = self.optimized_generator.compute_gradients(
            loss_g, tv_gen)
        accum_dis_ops = [
            self.accum_vars_dis[i].assign_add(
                tf.scalar_mul(const, gradient_variable[0]))
            for i, gradient_variable in enumerate(gradient_dis)
        ]
        accum_gen_ops = [
            self.accum_vars_gen[i].assign_add(
                tf.scalar_mul(const, gradient_variable[0]))
            for i, gradient_variable in enumerate(gradient_gen)
        ]

        train_step_dis = self.optimizer_discriminator.apply_gradients(
            zip(self.accum_vars_dis, tv_dis))
        train_step_gen = self.optimized_generator.apply_gradients(
            zip(self.accum_vars_gen, tv_gen))

        summary_dis = tf.summary.scalar('discriminator_loss', loss_d)
        summary_gen = tf.summary.scalar('generator_loss', loss_g)

        return train_step_dis, train_step_gen, accum_dis_ops, accum_gen_ops, loss_d, loss_g, summary_dis, summary_gen

    def train_epochs(self,
                     loss,
                     epochs=1,
                     iterations_per_epoch=500,
                     starting_epoch=0,
                     metrics_list=[],
                     names=[]):
        train_step_dis, train_step_gen, accum_dis_ops, accum_gen_ops, loss_discriminator, loss_generator, summary_dis, summary_gen = self.initialize_train_operations(
            loss)
        self.session.run(tf.global_variables_initializer())
        if self.resolution > 4:
            alpha_operation_gen = self.generator.return_assign_alpha_weight_operation(
            )
            alpha_operation_dis = self.discriminator.return_assign_alpha_weight_operation(
            )
        if len(names) != 0:
            self.generator.load_mapper(directories.get_mappers_folder() +
                                       names[0])
            self.generator.load_generator(directories.get_generators_folder() +
                                          names[1])

            self.discriminator.load(directories.get_discriminators_folder() +
                                    names[2])

        step_np = self.session.run(self.generator.get_step())
        step_increment = self.generator.increment_step()

        for epoch in range(starting_epoch, epochs):
            progressbar = tqdm(range(iterations_per_epoch))
            loss_dis = 0.0
            loss_gen = 0.0
            sum_loss_gen = 0.0
            sum_loss_dis = 0.0
            for iteration in progressbar:
                try:
                    self.session.run([self.zero_ops_dis, self.zero_ops_gen])

                    for n in range(config.n_generator):
                        for i in range(config.minibatches_per_apply):
                            _, loss_gen, sum_gen = self.session.run(
                                [accum_gen_ops, loss_generator, summary_gen])
                        self.session.run([train_step_gen])
                    for n in range(config.n_critic):
                        for i in range(config.minibatches_per_apply):
                            _, loss_dis, sum_dis = self.session.run([
                                accum_dis_ops, loss_discriminator, summary_dis
                            ])
                        self.session.run([train_step_dis])
                    sum_loss_dis += np.average(loss_dis)
                    sum_loss_gen += np.average(loss_gen)
                    self.writer.add_summary(sum_dis, step_np)
                    self.writer.add_summary(sum_gen, step_np)
                    step_np += 1
                    self.session.run(step_increment)
                    if self.resolution > 4:
                        self.session.run(
                            [alpha_operation_gen, alpha_operation_dis])
                    progressbar.set_description(
                        'ep:{} dis:{:2f}, gen:{:2f}'.format(
                            epoch, sum_loss_dis / (iteration + 1),
                            sum_loss_gen / (iteration + 1)))
                    progressbar.refresh()

                except tf.errors.OutOfRangeError:
                    break
            metrics_name_avg_std = self.evaluate_metrics(metrics_list)
            self.log_metrics(metrics_name_avg_std, epoch)
            names_to_ret = self.saving_procedure(epoch)
        return names_to_ret

    def log_metrics(self, metrics_name_score, epoch):
        for name_score in metrics_name_score:
            name = name_score[0]
            score_avg = name_score[1]
            score_std = name_score[2]
            self.writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(tag=name + ' average with resolution ' +
                                     str(self.resolution),
                                     simple_value=score_avg)
                ]), epoch)
            self.writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(tag=name + ' deviation with resolution ' +
                                     str(self.resolution),
                                     simple_value=score_std)
                ]), epoch)

    def evaluate_metrics(self, metrics_list):
        name_avg_std_score = []

        for metric in metrics_list:

            score_avg, score_std = metric.evaluate(
                self.session, self.generator, real_data=self.from_data_set)
            name = metric.get_name()
            name_avg_std_score.append((name, score_avg, score_std))
        return name_avg_std_score

    def create_few_images_from_dataset(self):
        from_data_set = self.data_generator.get_batch()
        from_data_set = self.session.run(from_data_set)
        self.image_generator.save_image(image_data=from_data_set,
                                        resolution=self.resolution,
                                        cols=4)

    def saving_procedure(self,
                         epoch,
                         num_images=config.batch_size,
                         save_images=True,
                         save_max_files=5):
        used_to_save = epoch % save_max_files
        name_generator = self.generator.save_generator(epoch=used_to_save)
        name_mapper = self.generator.save_mapper(epoch=used_to_save)
        name_discriminator = self.discriminator.save(epoch=used_to_save)
        if save_images:
            self.image_generator.save_image(self.session.run(
                self.generator.call_without_input()),
                                            cols=4,
                                            epochs=epoch,
                                            resolution=self.resolution,
                                            image_name='image-e{}-r{}.png')
        return [name_mapper, name_generator, name_discriminator]

    def initialize_dataset(self, dataset_name):
        self.data_generator = DataReader(dataset_name,
                                         self.resolution,
                                         shuffle_mb=1024)
