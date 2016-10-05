import tensorflow as tf
import prettytensor as pt
from infogan.misc.custom_ops import leaky_rectify

# class DiscrimNetworks(object):
#     def __init__(self, shared_template=None, encoder_template=None, image_shape=None, is_reg=False, ):
#         self._shared_template = shared_template
#         self._encoder_template = encoder_template
#         self.is_reg = is_reg
#         self.image_shape = image_shape
#         return
#
#     @property
#     def shared_template(self):
#         return self._shared_template
#
#     @property
#     def encoder_template(self):
#         return self.encoder_template

class InfoGAN_MNIST_net():
    def __init__(self, image_shape=28, is_reg=False, encoder_dim=None):
        self.image_shape = image_shape
        self.is_reg = is_reg
        self.encoder_dim = encoder_dim
        self._shared_template = self.infoGAN_mnist_shared_net()
        if self.is_reg == True:
            self._encoder_template = self.infoGAN_mnist_encoder_net()
        else:
            self._encoder_template = None

    @property
    def shared_template(self):
        return self._shared_template

    @property
    def encoder_template(self):
        return self._encoder_template

    def infoGAN_mnist_shared_net(self):
        shared_template = \
            (pt.template("input").
             reshape([-1] + list(self.image_shape)).
             custom_conv2d(64, k_h=4, k_w=4).
             apply(leaky_rectify).
             custom_conv2d(128, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(1024).
             fc_batch_norm().
             apply(leaky_rectify))
        return shared_template

    def infoGAN_mnist_encoder_net(self):
        encoder_template = \
            (self.shared_template.
             custom_fully_connected(128).
             fc_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(self.encoder_dim))
        return encoder_template

class dcgan_net():
    def __init__(self, image_shape=64, is_reg=False, encoder_dim=None):
        self.df_dim = 64
        self.image_shape = image_shape
        self._shared_template = self.dcgan_shared_net()
        self.is_reg = is_reg
        self.encoder_dim = encoder_dim

        if self.is_reg:
            self._encoder_template = self.dcgan_encoder_net()
        else:
            self._encoder_template = None

    @property
    def shared_template(self):
        return self._shared_template

    @property
    def encoder_template(self):
        return self._encoder_template

    def dcgan_shared_net(self):
        shared_template = \
            (pt.template("input").
             reshape([-1] + list(self.image_shape)).
             custom_conv2d(self.df_dim, name='d_h0_conv').
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim*2, name='d_h1_conv').
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim*4, name='d_h2_conv').
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim*8, name='d_h0_conv').
             conv_batch_norm().
             apply(leaky_rectify))
        return shared_template

    def dcgan_encoder_net(self):
        encoder_template = \
            (self._shared_template.
             custom_fully_connected(512).
             fc_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(self.encoder_dim))
        return encoder_template
