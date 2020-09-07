import argparse
import numpy as np


class ArgumentParser():
    def __init__(self, arguments):
        command_line_parameters = arguments

        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--generator', required=False)
        parser.add_argument('-d', '--discriminator', required=False)
        parser.add_argument('-r',
                            '--resolution',
                            required=False,
                            type=int,
                            default=8)
        parser.add_argument('-gc',
                            '--googlecolab',
                            required=False,
                            action='store_true')
        parser.add_argument('-le',
                            '--load_epoch',
                            required=False,
                            action='store_true')
        parser.add_argument('-e',
                            '--epoch',
                            required=False,
                            type=int,
                            default=0)
        parser.add_argument('-nr',
                            '--newresolution',
                            required=False,
                            action='store_true')
        parser.add_argument('-nm',
                            '--nometric',
                            required=False,
                            action='store_true')
        parser.add_argument('-nb',
                            '--number_batches',
                            required=False,
                            type=int,
                            default=15)
        parser.add_argument('-pro',
                            '--progan',
                            required=False,
                            action='store_true')
        self.arguments = parser.parse_args()

    def is_google_colab(self):
        return self.arguments.googlecolab

    def should_load_epoch(self):
        return self.arguments.load_epoch

    def is_progan(self):
        return self.arguments.progan

    def get_starting_epoch(self):
        if self.arguments.newresolution:
            return 0
        if self.should_load_epoch():
            return self.arguments.epoch + 1
        return self.arguments.epoch

    def get_resolution_log2(self):
        return int(np.math.log2(self.arguments.resolution))

    def use_metrics(self):
        return not self.arguments.nometric

    def get_resolution(self):
        return self.arguments.resolution

    def get_default_load_model_names(self, default_name=None):
        if self.arguments.load_epoch and default_name is not None:
            res = self.get_resolution()
            if self.arguments.newresolution:
                res = res // 2
            return [
                default_name.format(self.arguments.epoch, res),
                default_name.format(self.arguments.epoch, res),
                default_name.format(self.arguments.epoch, res)
            ]
        return []
