from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
import infogan.models.generative_networks as G
import infogan.models.discriminative_networks as D


class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, is_reg, batch_size, image_shape, network_type):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.is_reg = is_reg
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        self.keys = ['prob', 'logits', 'features']

        if self.is_reg:
            self.encoder_dim=self.reg_latent_dist.dist_flat_dim
            self.keys = self.keys + ['reg_dist_info']
        else:
            self.encoder_dim=None

        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        self.set_D_net()
        self.set_G_net()

    def set_D_net(self):
        with tf.variable_scope("d_net"):
            if self.network_type == "mnist":
                self.D_model = D.InfoGAN_MNIST_net(image_shape=self.image_shape,is_reg=self.is_reg,encoder_dim=self.encoder_dim)
                shared_template = self.D_model.shared_template
                self.discriminator_template = shared_template.custom_fully_connected(1)
                self.encoder_template = self.D_model.encoder_template
            else:
                if self.network_type == 'dcgan':
                    self.D_model = D.dcgan_net(image_shape=self.image_shape,is_reg=self.is_reg,encoder_dim=self.encoder_dim)
                elif self.network_type == 'deeper_dcgan':
                    self.D_model = D.deeper_dcgan_net(image_shape=self.image_shape,is_reg=self.is_reg,encoder_dim=self.encoder_dim)
                else:
                    raise NotImplementedError
                self.shared_template = self.D_model.shared_template
                self.discriminator_template = self.shared_template.custom_fully_connected(1)
                self.encoder_template = self.D_model.encoder_template


    def set_G_net(self):
        with tf.variable_scope("g_net"):
            if self.network_type == 'mnist':
                self.gen_model = G.InfoGAN_mnist_net()
                self.generator_template = self.gen_model.infoGAN_mnist_net(self.image_shape)
            else:
                if self.network_type == 'dcgan':
                    self.gen_model = G.dcgan_net()
                elif self.network_type == 'deeper_dcgan':
                    self.gen_model = G.deeper_dcgan_net()
                self.generator_template = self.gen_model.gen_net(self.image_shape)
                else:
                    raise NotImplementedError

    def discriminate(self, x_var):
        d_features = self.shared_template.construct(input=x_var)
        d_logits = self.discriminator_template.construct(input=x_var)[:,0]
        d_prob = tf.nn.sigmoid(d_logits)

        d_dict = dict.fromkeys(self.keys)

        d_dict['features'] = d_features
        d_dict['logits'] = d_logits
        d_dict['prob'] = d_prob

        if self.is_reg:
            reg_dist_flat = self.encoder_template.construct(input=x_var)
            reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
            d_dict['reg_dist_info'] = reg_dist_info

        return d_dict

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        if self.network_type=="mnist":
            x_dist_info = self.output_dist.activate_dist(x_dist_flat)
            return self.output_dist.sample(x_dist_info), x_dist_info
        else:
            return x_dist_flat, None

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)
