import metrics.metrics as metrics
import metrics.inception_processing as inception_processing
import utilities.directories as directories
import training.config as config
import numpy as np
from tqdm import tqdm
import utilities.directories as directories


class FrechettInceptionDistance:
    name = 'frechett inception distance'
    number_of_batches_for_one_file = 4

    def prepare_data_for_metric(self, session, generator, real_data,
                                number_of_batches):
        for i in range(number_of_batches //
                       self.number_of_batches_for_one_file):
            fake_list = []
            for _ in range(self.number_of_batches_for_one_file):
                fake_list.append(session.run(generator.call_without_input()))
            fake = np.asarray(fake_list)
            np.save(
                directories.get_testing_folder() + 'fake' + str(i) + '.npy',
                fake)
            real_list = []
            for _ in range(self.number_of_batches_for_one_file):
                real_list.append(session.run(real_data))
            real = np.asarray(real_list)
            np.save(
                directories.get_testing_folder() + 'real' + str(i) + '.npy',
                real)

    def evaluate(self,
                 session,
                 generator,
                 real_data,
                 truncation=1.0,
                 number_activations_in_one_metric_calculation=1024,
                 number_of_activations_batches=10):
        number_of_batches = number_activations_in_one_metric_calculation // config.batch_size

        score_list = []
        progressbar_frechett_calculations = tqdm(
            range(number_of_activations_batches))
        activations_list1 = []
        activations_list2 = []
        generate_data_op = generator.truncation_call(truncation)
        for batch in progressbar_frechett_calculations:
            for i in range(number_of_batches):
                data1 = session.run(generate_data_op)
                data2 = session.run(real_data)
                preprocessed_data1 = inception_processing.preprocess_for_inception(
                    data1)
                preprocessed_data2 = inception_processing.preprocess_for_inception(
                    data2)
                activations1 = inception_processing.predict_inception(
                    preprocessed_data1)
                activations2 = inception_processing.predict_inception(
                    preprocessed_data2)
                activations_list1.append(activations1)
                activations_list2.append(activations2)

            activations_from_list1 = np.concatenate(activations_list1)
            activations_from_list2 = np.concatenate(activations_list2)
            activations_list1.clear()
            activations_list2.clear()
            score = metrics.calculate_frechett(activations_from_list1,
                                               activations_from_list2)

            info = 'Calculated score is ' + str(score) + ' ' + str(truncation)
            self.append_file_info(info)
            print(info)
            score_list.append(score)
        np_scores = np.asarray(score_list)
        avg, std = np.mean(np_scores), np.std(np_scores)
        print(self.name + ' avg: {}  std: {} '.format(avg, std))
        self.append_file(avg, std)
        return avg, std

    def append_file(self, avg, std):
        with open(directories.scores_file, "a") as myfile:
            myfile.write(self.name + ' avg: {}  std: {} \n'.format(avg, std))

    def append_file_info(self, info):
        with open(directories.scores_file, "a") as myfile:
            myfile.write(info + '\n')

    def get_name(self):
        return self.name