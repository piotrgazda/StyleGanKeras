import tensorflow as tf
import numpy as np


def _blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x,
                               f,
                               strides=strides,
                               padding='SAME',
                               data_format='NCHW')
    x = tf.cast(x, orig_dtype)
    return x


def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None
                                      for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def _downscale2d(x, factor=2, gain=1):

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x,
                          ksize=ksize,
                          strides=ksize,
                          padding='VALID',
                          data_format='NCHW')


def blur2d(x, f=[1, 2, 1], normalize=True):
    with tf.variable_scope('Blur2D'):

        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)

            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)

            return y, grad

        return func(x)


def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):

        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)

            return y, grad

        return func(x)


def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):

        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1 / factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)

            return y, grad

        return func(x)


def nf(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4  # NCHW
    x -= tf.math.reduce_mean(x, axis=[2, 3], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    x *= tf.math.rsqrt(
        tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
    return x


def apply_noise(x, weight, noise_var=None, randomize_noise=True):
    assert len(x.shape) == 4  # NCHW
    if noise_var is None or randomize_noise:
        noise = tf.random.normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]],
                                 dtype=tf.float32)
    else:
        noise = tf.cast(noise_var, x.dtype)

    return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])


def pixel_norm(x, epsilon=1e-8):
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.math.rsqrt(
        tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def style_mod(x, dlatent, bias, weights, **kwargs):
    style = tf.matmul(dlatent, weights) + bias
    style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
    return x * (style[:, 0] + 1) + style[:, 1]
