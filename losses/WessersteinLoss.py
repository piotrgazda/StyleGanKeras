import tensorflow as tf


class WessersteinLossGPNonsaturating:
    def __init__(self, penalty_weight):
        self.penalty_weight = penalty_weight

    def evaluate(self, generator, discriminator, real_features):
        real_logit = discriminator(real_features)

        fake_features = generator.call_without_input()
        fake_logit = discriminator(fake_features)

        d_loss_gan = tf.nn.softplus(fake_logit) + tf.nn.softplus(-real_logit)
        real_loss = tf.reduce_sum(real_logit)
        real_grads = tf.gradients(real_loss, [real_features])[0]
        r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        d_loss = d_loss_gan + r1_penalty * (self.penalty_weight * 0.5)
        d_loss = tf.reduce_mean(d_loss)

        g_loss = tf.nn.softplus(-fake_logit)
        g_loss = tf.reduce_mean(g_loss)

        return g_loss, d_loss
