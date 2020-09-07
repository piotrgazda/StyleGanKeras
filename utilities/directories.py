import training.config as config
import numpy as np
import datetime

root_folder = '.'
models_save_folder = 'models-save/'
images_save_folder = 'images-generated/'
logs_save_folder = 'logs/'
testing_data_folder = 'data_for_testing/'
dataset_directory = ''
dataset_directory_in_gdrive = '/content/gdrive/My Drive/ffhq-dataset/tfrecords/ffhq/ffhq-r{0:0=2d}.tfrecords'
current_time_logs = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
scores_file = 'scores.txt'


def get_generators_folder():
    return models_save_folder + 'generators/'


def get_discriminators_folder():
    return models_save_folder + 'discriminators/'


def get_mappers_folder():
    return models_save_folder + 'mappers/'


def correct_for_google_colab(is_google_colab):
    if is_google_colab:
        global root_folder, models_save_folder, images_save_folder, logs_save_folder, dataset_directory, testing_data_folder, scores_file

        root_folder = '/content/gdrive/My Drive/'
        models_save_folder = root_folder + 'stylegankeras/' + models_save_folder
        images_save_folder = root_folder + 'stylegankeras/' + images_save_folder
        logs_save_folder = root_folder + 'stylegankeras/' + logs_save_folder
        dataset_directory = '/content/gdrive/My Drive/ffhq-dataset/tfrecords/ffhq/' + dataset_directory
        testing_data_folder = '.'
        scores_file = root_folder + 'stylegankeras/' + scores_file


def get_model_path():
    return models_save_folder


def get_ffhq_dataset_path_in_gdrive(resolution,
                                    name='ffhq-r{0:0=2d}.tfrecords'):
    return dataset_directory + name.format(int(np.math.log2(resolution)))


def append_formatted_paths(directory, resolution=None):
    if resolution is not None:
        return directory + 'r{}/'.format(resolution)
    else:
        return directory


def format_name_with_resolution_epochs(name, resolution, epochs):
    name_to_return = name
    if resolution is not None:
        name_to_return = name_to_return.format(resolution)
    if epochs is not None:
        name_to_return = name_to_return.format(epochs)
    return name_to_return


def get_path_for_images(image_name, resolution=None, epochs=None):
    if image_name is None:
        image_name = 'default_image-e{}-r{}.png'
    if resolution is not None and epochs is not None:
        image_name = image_name.format(epochs, resolution)
    return images_save_folder + image_name


def get_testing_folder():
    return testing_data_folder


def get_logs_folder():
    return logs_save_folder


def get_logs_full_filename(resolution):
    return logs_save_folder + current_time_logs + '-r' + str(resolution)
