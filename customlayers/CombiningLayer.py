from tensorflow.keras.layers import Layer
import tensorflow as tf
import training.config as config
import customlayers.ops as ops


class CombiningLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super(CombiningLayer, self).build(input_shape)
        self.lerp_weight = self.add_weight(name='lerp_weight',
                                           shape=[],
                                           trainable=False,
                                           initializer=tf.initializers.zeros(),
                                           dtype=tf.float32)
      

    def call(self, input):
        beforeblock = input[0]
        afterblock = input[1]
        result = tf.cond(
            tf.less(self.lerp_weight,
                    1.0), lambda: beforeblock + self.lerp_weight *
            (afterblock - beforeblock), lambda: afterblock)

        return result

    def return_assign_alpha_weight_operation(self):
        return tf.cond(
            tf.less(self.lerp_weight, 1.0), lambda: tf.assign(
                self.lerp_weight, self.lerp_weight + config.
                combining_val_increment), lambda: self.lerp_weight)
    def return_alpha(self):
        return self.lerp_weight