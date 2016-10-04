import tensorflow as tf
import prettytensor as pt
from infogan.misc.custom_ops import leaky_rectify

class DiscrimNetworks(object):
    def __init__(self):
        print "Creating discrim networks"

    def infoGAN_mnist_shared_net(self, image_shape):
        shared_template = \
            (pt.template("input").
             reshape([-1] + list(image_shape)).
             custom_conv2d(64, k_h=4, k_w=4).
             apply(leaky_rectify).
             custom_conv2d(128, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(1024).
             fc_batch_norm().
             apply(leaky_rectify))
        return shared_template

    def infoGAN_mnist_encoder_net(self, shared_template, encoder_dim):
        encoder_template = \
            (shared_template.
             custom_fully_connected(128).
             fc_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(encoder_dim))
        return encoder_template
