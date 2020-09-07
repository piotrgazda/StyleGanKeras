from PIL import Image
import training.config as config
import numpy as np
import tensorflow as tf
import utilities.directories as directories


class ImageOutGenerator:
    def __init__(self,
                 image_saving_path=config.images_save_folder,
                 data_format='NCHW'):

        self.image_saving_path = image_saving_path
        self.data_format = data_format
        self.iterations = 0
        self.image_saving_path = image_saving_path

    def get_pillow_image(self, image_data, resolution, cols):
        y = image_data
        if self.data_format == 'NCHW':
            y = np.rollaxis(y, 1, 4)
        y = np.clip(y, -1.0, 1.0)
        y = y.reshape(-1, resolution, 3)
        y = np.array_split(y, cols, axis=0)
        y = np.concatenate(y, axis=1)
        y = y + 1.0
        y = y * 127.5
        img = Image.fromarray(np.uint8(y))
        return img

    def generate_truncation_diagram(self,
                                    mapper,
                                    generator,
                                    resolution,
                                    session,
                                    truncation_step=0.25):
        random = tf.random.normal(shape=(1, config.latent_size))
        styles = []
        truncation_value = 1.0
        while truncation_value >= -1.0:
            styles.append(
                session.run(
                    mapper.truncation_call(random, random, truncation_value)))
            truncation_value -= truncation_step
        images = session.run(generator(np.concatenate(styles)))
        images_split = np.array_split(images,
                                      2.0 // truncation_step + 1,
                                      axis=0)
        images_concatenated = np.concatenate(images_split, axis=3)
        images_concatenated = np.reshape(images_concatenated,
                                         newshape=(3, resolution, -1))
        if self.data_format == 'NCHW':
            y = np.rollaxis(images_concatenated, 0, 3)
        y = np.clip(y, -1.0, 1.0)
        y = y + 1.0
        y = y * 127.5

        img = Image.fromarray(np.uint8(y))

        return img

    def generate_truncation_diagram_eager(self,
                                          mapper,
                                          generator,
                                          resolution,
                                          truncation_step=0.25,
                                          convert_to_pillow=True):
        random = tf.random.normal(shape=(1, config.latent_size))
        styles = []
        truncation_value = 1.0
        while truncation_value >= -1.0:
            styles.append(
                mapper.truncation_call(random, random,
                                       truncation_value).numpy())
            truncation_value -= truncation_step
        images = generator(np.concatenate(styles)).numpy()
        images_split = np.array_split(images,
                                      2.0 // truncation_step + 1,
                                      axis=0)
        images_concatenated = np.concatenate(images_split, axis=3)
        images_concatenated = np.reshape(images_concatenated,
                                         newshape=(3, resolution, -1))
        if self.data_format == 'NCHW':
            y = np.rollaxis(images_concatenated, 0, 3)
        y = np.clip(y, -1.0, 1.0)
        y = y + 1.0
        y = y * 127.5
        if convert_to_pillow:
            y = Image.fromarray(np.uint8(y))

        return y

    def generate_style_mix_image(self,
                                 mapper,
                                 generator,
                                 resolution,
                                 session,
                                 truncation=1.0,
                                 cols=4,
                                 rows=4):
        style_mixing_place_iterator = 0
        style_mixing_place_iterator_increment = 2

        rows = ((int(np.math.log2(resolution)) * 2 - 2) // 2) - 1

        random = np.random.normal(size=(cols + rows, config.latent_size))
        one_styled_styles = mapper(random, random)
        one_styled_styles = session.run(one_styled_styles)
        one_styled_styles = one_styled_styles[:cols + rows]
        styles_first_row = one_styled_styles[:cols]
        styles_first_col = one_styled_styles[cols:]
        one_styled_faces = generator(one_styled_styles)
        one_styled_faces = session.run(one_styled_faces)

        black_square = np.zeros(shape=(1, 3, resolution, resolution))

        faces_first_row = one_styled_faces[:cols]
        faces_first_col = one_styled_faces[cols:]
        rows_list = []
        rows_list.append(np.concatenate([black_square, faces_first_row]))

        for i in range(rows):
            style_mixing_place_iterator += style_mixing_place_iterator_increment
            style_mixed = np.concatenate([
                np.tile(styles_first_col[i, 0:style_mixing_place_iterator],
                        reps=[cols, 1, 1]),
                styles_first_row[:, style_mixing_place_iterator:]
            ],
                                         axis=1)
            rows_list.append(
                np.concatenate([
                    np.reshape(faces_first_col[i],
                               newshape=(1, 3, resolution, resolution)),
                    session.run(generator(style_mixed))
                ]))

        for i in range(rows + 1):
            rows_list[i] = np.concatenate(np.array_split(
                rows_list[i], cols + 1),
                                          axis=3)
        y = np.concatenate(rows_list)
        if self.data_format == 'NCHW':
            y = np.rollaxis(y, 1, 4)
        y = np.clip(y, -1.0, 1.0)
        y = y.reshape(-1, resolution * (cols + 1), 3)
        y = y + 1.0
        y = y * 127.5

        img = Image.fromarray(np.uint8(y))

        return img

    def generate_style_mix_image_eager(self,
                                       mapper,
                                       generator,
                                       resolution,
                                       column_orientation=True,
                                       truncation=1.0,
                                       cols=4,
                                       rows=4):
        style_mixing_place_iterator = 0
        style_mixing_place_iterator_increment = 1

        random = np.random.normal(size=(cols + rows, config.latent_size))
        one_styled_styles = mapper.truncation_call(random, random,
                                                   truncation).numpy()
        one_styled_styles = one_styled_styles[:cols + rows]
        styles_first_row = one_styled_styles[:cols]
        styles_first_col = one_styled_styles[cols:]
        one_styled_faces = generator(one_styled_styles).numpy()

        black_square = np.zeros(shape=(1, 3, resolution, resolution))

        faces_first_row = one_styled_faces[:cols]
        faces_first_col = one_styled_faces[cols:]
        rows_list = []
        rows_list.append(np.concatenate([black_square, faces_first_row]))

        for i in range(rows):
            style_mixing_place_iterator += style_mixing_place_iterator_increment
            if column_orientation:
                style_mixed = np.concatenate([
                    np.tile(styles_first_col[i, 0:style_mixing_place_iterator],
                            reps=[cols, 1, 1]),
                    styles_first_row[:, style_mixing_place_iterator:]
                ],
                                             axis=1)
                rows_list.append(
                    np.concatenate([
                        np.reshape(faces_first_col[i],
                                   newshape=(1, 3, resolution, resolution)),
                        generator(style_mixed).numpy()
                    ]))
            else:
                style_mixed = np.concatenate([
                    styles_first_row[:, 0:style_mixing_place_iterator],
                    np.tile(styles_first_col[i, style_mixing_place_iterator:],
                            reps=[cols, 1, 1])
                ],
                                             axis=1)
                rows_list.append(
                    np.concatenate([
                        np.reshape(faces_first_col[i],
                                   newshape=(1, 3, resolution, resolution)),
                        generator(style_mixed).numpy()
                    ]))

        for i in range(rows + 1):
            rows_list[i] = np.concatenate(np.array_split(
                rows_list[i], cols + 1),
                                          axis=3)
        y = np.concatenate(rows_list)
        if self.data_format == 'NCHW':
            y = np.rollaxis(y, 1, 4)
        y = np.clip(y, -1.0, 1.0)
        y = y.reshape(-1, resolution * (cols + 1), 3)
        y = y + 1.0
        y = y * 127.5

        img = Image.fromarray(np.uint8(y))

        return img

    def save_image(self,
                   image_data,
                   resolution,
                   cols,
                   image_name=None,
                   epochs=None):
        img = self.get_pillow_image(image_data=image_data,
                                    resolution=resolution,
                                    cols=cols)
        full_name = directories.get_path_for_images(image_name=image_name,
                                                    epochs=epochs,
                                                    resolution=resolution)
        
        img.save(full_name)

    def save_image_as_numpy(self, full_filename, session, generator):
        images_op = generator.call_without_input()
        images_batch = session.run([images_op])
        np.save(full_filename, images_batch)

    def save_images_as_numpy(self, session, generator, number_of_batches):
        for i in range(number_of_batches):
            print('Generating numpy array of images nr ' + str(i))
            self.save_image_as_numpy(
                directories.get_testing_folder() + str(i) + '.npy', session,
                generator)
