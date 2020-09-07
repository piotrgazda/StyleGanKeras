import metrics.metrics as metrics
import metrics.inception_processing as inception_processing
import utilities.directories as directories
import training.config as config
import numpy as np
from tqdm import tqdm
import tensorflow as tf

class DiscriminativeMetric:
    name = 'discriminative metric'
    number_of_batches_for_one_file = 4

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def evaluate(self,
                 session,
                 generator,
                 real_data,
                 number_activations_in_one_metric_calculation=1024,
                 number_of_activations_batches=10):
        number_of_batches = number_activations_in_one_metric_calculation // config.batch_size

        score_list_real = []
        score_list_fake = []

        progressbar_frechett_calculations = tqdm(
            range(number_of_activations_batches))
        real_logits = []
        fake_logits = []
        fake_logit_op = self.discriminator(generator.call_without_input())
        real_data = tf.reshape(real_data,
                                        shape=(config.batch_size, 3,
                                               generator.resolution,
                                               generator.resolution))       
        real_logit_op = self.discriminator(real_data)
        for batch in progressbar_frechett_calculations:
            for i in range(number_of_batches):
                fake_logit = session.run(fake_logit_op)
                real_logit = session.run(real_logit_op)
                fake_logits.append(fake_logit)
                real_logits.append(real_logit)

            score_real = np.mean(np.concatenate(real_logits))
            score_fake = np.mean(np.concatenate(fake_logits))
            fake_logits.clear()
            real_logits.clear()  
            print('Calculated score for real is ' + str(score_real) + 'for fake is '+ str(score_fake))
            score_list_real.append(score_real)
            score_list_fake.append(score_fake)
        
        np_scores_real = np.asarray(score_list_real)
        np_scores_fake = np.asarray(score_list_fake)

        avg_real, std_real = np.mean(np_scores_real), np.std(np_scores_real)
        avg_fake, std_fake = np.mean(np_scores_fake), np.std(np_scores_fake)

        print('real ' + self.name + ' avg: {}  std: {}'.format(avg_real, std_real))
        print('fake ' + self.name + ' avg: {}  std: {}'.format(avg_fake, std_fake))

        return avg_fake, std_fake

    def get_name(self):
        return self.name