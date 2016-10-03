from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
import infogan.misc.datasets as datasets
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, imagenet]")
flags.DEFINE_string("output_size", 64, "Size of the images to generate")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/" + FLAGS.dataset
    root_checkpoint_dir = "ckt/" + FLAGS.dataset
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 50

    exp_name = "%s_%s" % (FLAGS.dataset, timestamp)

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    if FLAGS.dataset == "mnist":
        dataset = datasets.MnistDataset()
    elif FLAGS.dataset == 'imagenet':
        dataset = datasets.ImageNetDatset(output_size=output_size)
    elif FLAGS.dataset == 'celebA':
        dataset = datasets.celebADataset()

    latent_spec = [
        (Uniform(100), False)]
        # (Categorical(10), True),
        # (Uniform(1, fix_std=True), True),
        # (Uniform(1, fix_std=True), True),
    # ]

    is_reg = False
    for x,y in latent_spec:
        if y:
            is_reg = True

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        is_reg=is_reg,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
