import metrics.metrics as metrics
import metrics.inception_processing as inception_processing
import utilities.directories as directories
import training.config as config
import numpy as np
from tqdm import tqdm


class GeometryScore:
    name = 'geometry score'

    def evaluate(self,
                 session,
                 generator,
                 real_data,
                 number_activations_in_one_metric_calculation=10000,
                 number_of_activations_batches=10):
        number_of_batches = number_activations_in_one_metric_calculation // config.batch_size

        score_list = []
        progressbar_frechett_calculations = tqdm(
            range(number_of_activations_batches))
        fake_data_list = []
        real_data_list = []
        for batch in progressbar_frechett_calculations:
            for i in range(number_of_batches):
                data1 = session.run(generator.call_without_input())
                data2 = session.run(real_data)
                data1 = np.clip(data1, -1.0, 1.0)
                data1 = np.reshape(
                    data1,
                    (-1, generator.resolution * generator.resolution * 3))

                data2 = np.reshape(
                    data2,
                    (-1, generator.resolution * generator.resolution * 3))
                fake_data_list.append(data1)
                real_data_list.append(data2)

            fakes = np.concatenate(fake_data_list)
            reals = np.concatenate(real_data_list)

            fake_data_list.clear()
            real_data_list.clear()

            rlts_fake = metrics.rlts(fakes)
            rlts_real = metrics.rlts(reals)
            score = metrics.geom_score(rlts_fake, rlts_real)
            print('Calculated score is ' + str(score))
            score_list.append(score)
        np_scores = np.asarray(score_list)
        avg, std = np.mean(np_scores), np.std(np_scores)
        print(self.name + ' avg: {}  std: {}'.format(avg, std))
        return avg, std

    def get_name(self):
        return self.name