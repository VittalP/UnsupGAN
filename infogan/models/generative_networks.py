import tensorflow as tf
import prettytensor as pt

class GenNetworks(object):
    def __init__(self):
        print "__init__ - ing"

class InfoGAN_mnist_net():
    
    def infoGAN_mnist_net(self, image_shape):
        image_size = image_shape[0]
        generator_template = \
            (pt.template("input").
             custom_fully_connected(1024).
             fc_batch_norm().
             apply(tf.nn.relu).
             custom_fully_connected(image_size / 4 * image_size / 4 * 128).
             fc_batch_norm().
             apply(tf.nn.relu).
             reshape([-1, image_size / 4, image_size / 4, 128]).
             custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
             flatten())
        return generator_template
