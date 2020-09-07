from metrics.FrechettInceptionDistance import FrechettInceptionDistance
from metrics.InceptionScore import InceptionScore
import utilities.directories as directories
import os
from losses.WessersteinLoss import WessersteinLossGPNonsaturating
import training.config as config
from shutil import copyfile


def get_metrics_list(use_metrics):
    if use_metrics:
        return [InceptionScore(), FrechettInceptionDistance()]
    else:
        return []


def mount_google_disc(is_google_colab):
    if is_google_colab:
        from google.colab import drive
        drive.mount('/content/gdrive')
    directories.correct_for_google_colab(is_google_colab)


def __print_copying_message():
    print('...')
    print('...')
    print('...')
    print('Copying dataset... might take time...')
    print('...')
    print('...')
    print('...')


def copy_dataset(destination_copy, resolution, is_google_colab):
    if not os.path.isfile(destination_copy) and is_google_colab:
        __print_copying_message()
        copyfile(
            directories.get_ffhq_dataset_path_in_gdrive(resolution=resolution),
            destination_copy)


def get_loss():
    return WessersteinLossGPNonsaturating(penalty_weight=config.lambdapar)
