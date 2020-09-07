import utilities.directories as directories
from utilities.ArgumentParser import ArgumentParser
import sys
import os
import utilities.utils as utils


def prepare_folders(is_google_colab):
    utils.mount_google_disc(is_google_colab)
    os.makedirs(directories.get_discriminators_folder(), exist_ok=True)
    os.makedirs(directories.get_mappers_folder(), exist_ok=True)
    os.makedirs(directories.get_generators_folder(), exist_ok=True)
    os.makedirs(directories.images_save_folder, exist_ok=True)
    os.makedirs(directories.get_logs_folder(), exist_ok=True)


if __name__ == "__main__":

    parser = ArgumentParser(sys.argv)
    is_google_colab = parser.is_google_colab()
    prepare_folders(is_google_colab)
