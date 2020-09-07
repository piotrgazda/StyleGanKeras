from tensorflow.keras.layers import Layer
import tensorflow as tf


class MinibatchStdDev(Layer):
    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        self.group_size = group_size
        self.num_new_features = num_new_features
        super(MinibatchStdDev, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Be sure to call this at the end
        super(MinibatchStdDev, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2],
                input_shape[3])

    def call(self, x):
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        # [NCHW]  Input shape.
        s = x.shape
        # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.reshape(x, [
            self.group_size, -1, self.num_new_features,
            s[1] // self.num_new_features, s[2], s[3]
        ])
        # [GMncHW] Cast to FP32.
        y = tf.cast(y, tf.float32)
        # [GMncHW] Subtract mean over group.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        # [MncHW]  Calc variance over group.
        y = tf.reduce_mean(tf.square(y), axis=0)
        # [MncHW]  Calc stddev over group.
        y = tf.sqrt(y + 1e-8)
        # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        # [Mn11] Split channels into c channel groups
        y = tf.reduce_mean(y, axis=[2])
        # [Mn11]  Cast back to original data type.
        y = tf.cast(y, x.dtype)
        # [NnHW]  Replicate over group and pixels.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        # [NCHW]  Append as new fmap.
        return tf.concat([x, y], axis=1)
