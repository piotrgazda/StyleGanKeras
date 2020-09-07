import metrics.metrics as metrics
import metrics.inception_processing as inception_processing
import training.config as config
from utilities import directories
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class InceptionScore:
    name = 'inception score'
    number_of_batches_for_file = 4

    def prepare_data_for_metric(self,
                                session,
                                generator,
                                real_data=None,
                                number_of_batches=10):
        directory = directories.get_testing_folder()
        for i in range(number_of_batches // self.number_of_batches_for_file):
            data_list = []
            for _ in range(self.number_of_batches_for_file):
                data_list.append(session.run(generator.call_without_input()))
            data = np.asarray(data_list)
            np.save(directory + str(i) + '.npy', data)

    def evaluate(self,
                 session,
                 generator,
                 real_data=None,
                 number_samples_to_test=1024,
                 number_of_activations_batches=10):
        number_of_batches = number_samples_to_test // config.batch_size

        score_list = []
        activations_list = []
        progressbar_frechett_calculations = tqdm(
            range(number_of_activations_batches))
        for batch in progressbar_frechett_calculations:
            for i in range(number_of_batches):
                data = session.run(generator.call_without_input())
                preprocessed_data = inception_processing.preprocess_for_inception(
                    data)
                activations = inception_processing.predict_inception(
                    preprocessed_data)
                activations_list.append(activations)

            score = metrics.calculate_inception_score(
                np.concatenate(activations_list))
            print('Calculated score is ' + str(score))
            score_list.append(score)
            activations_list.clear()
        np_scores = np.asarray(score_list)
        avg, std = np.mean(np_scores), np.std(np_scores)
        print(self.name + ' avg: {}  std: {}'.format(avg, std))
        return avg, std

    def get_name(self):
        return self.name