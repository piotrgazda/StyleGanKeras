import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import tensorflow as tf

inception_model_with_top = None
inception_input_shape = (3, 299, 299)
inception_model_without_top = None

def load_inception_model():
    global inception_model
    global inception_model_without_top
    if inception_model is None:
        tf.keras.backend.set_image_data_format('channels_first')
        inception_model_with_top = InceptionV3(include_top=True, pooling='avg')
        inception_model_without_top =Model() 


def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


def preprocess_for_inception(images):
    images = np.clip(images, -1.0, 1.0)
    images = scale_images(images, new_shape=inception_input_shape)
    return images


def predict_inception(images):
    if inception_model is None:
        load_inception_model()
    return inception_model.predict(images)
