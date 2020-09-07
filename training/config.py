batch_size = 4
latent_size = 512  # latent size of Z vector
leaky_relu = 0.2  # leaky relu parameter
fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_max = 512  # Maximum number of feature maps in any layer.
number_of_images_shown_for_cycle = 1000000
iterations_per_epoch = 300
minibatches_per_apply = 1
combining_val_increment = 1 / (number_of_images_shown_for_cycle /
                               (batch_size * minibatches_per_apply))
epochs = (number_of_images_shown_for_cycle *
          2) // (iterations_per_epoch * batch_size * minibatches_per_apply) + 1
weights_clipping = 0.01
learning_rate = 0.001
adam_beta1 = 0.0
adam_beta2 = 0.99
n_critic = 1
n_generator = 1
lambdapar = 10.0
data_unformatted_name = 'ffhq-r{0:0=2d}.tfrecords'
image_unformatted_name = 'image-e{}-r{}.png'
image_unformatted_placeholder = 'image-number-{}.png'
models_save_folder = 'models-save/'
images_save_folder = 'images-generated/'
logs_save_folder = 'logs/'
testing_data_folder = 'data_for_testing/'
generators_folder = 'generators'
discriminator_folder = 'discriminator'
mapper_name = 'mapper-e{}-r{}.h5'
generator_name = 'generator-e{}-r{}.h5'
discriminator_name = 'discriminator-e{}-r{}.h5'
inception_directory = 'inception'
inception_name = 'inception_v3_features.pkl'
