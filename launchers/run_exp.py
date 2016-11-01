from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli

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
flags.DEFINE_string("train_dataset", "cifar", "The name of dataset in ./data")
flags.DEFINE_string("val_dataset", "cifar", "The name of dataset in ./data")
flags.DEFINE_integer("output_size", 64, "Size of the images to generate")
flags.DEFINE_integer("categories", None, "Size of the images to generate")
flags.DEFINE_integer("batch_size", 128, "Size of the images to generate")
flags.DEFINE_bool("train", True, "Training mode or testing mode")
flags.DEFINE_string("exp_name", None, "Used to load model")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/" + FLAGS.train_dataset
    root_checkpoint_dir = "ckt/" + FLAGS.train_dataset
    root_samples_dir = "samples/" + FLAGS.train_dataset
    batch_size = FLAGS.batch_size
    updates_per_epoch = 100
    max_epoch = 50

    if not FLAGS.exp_name:
        exp_name = "fc-t-%s_v-%s_o-%d" % (FLAGS.train_dataset, FLAGS.val_dataset,
                                       FLAGS.output_size)
        if FLAGS.categories is not None:
            exp_name = exp_name + "_c-%d" % (FLAGS.categories)
        exp_name = exp_name + "_%s" % (timestamp)
    else:
        exp_name = FLAGS.exp_name

    print("Experiment Name: %s" % (exp_name))

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    samples_dir = os.path.join(root_samples_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(samples_dir)

    output_dist = None
    network_type = 'dcgan'
    if FLAGS.train_dataset == "mnist":
        dataset = datasets.MnistDataset()
        output_dist = MeanBernoulli(dataset.image_dim)
        network_type = 'mnist'
        dataset.batch_idx = 100
    else:
        dataset = datasets.Dataset(name=FLAGS.train_dataset,
                                   batch_size=batch_size,
                                   output_size=FLAGS.output_size)
    val_dataset = datasets.Dataset(name=FLAGS.val_dataset,
                                   batch_size=batch_size,
                                   output_size=FLAGS.output_size)

    latent_spec = [
        (Uniform(100), False)
    ]
    if FLAGS.categories is not None:
        latent_spec.append((Categorical(FLAGS.categories), True))

    is_reg = False
    for x, y in latent_spec:
        if y:
            is_reg = True

    model = RegularizedGAN(
        output_dist=output_dist,
        latent_spec=latent_spec,
        is_reg=is_reg,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type=network_type,
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        isTrain=FLAGS.train,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        samples_dir=samples_dir,
        max_epoch=max_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=2e-3,
        discriminator_learning_rate=2e-3,
    )

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    algo.init_opt()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.train:
            algo.train(sess)
        else:
            restorer = tf.train.Saver()
            restorer.restore(sess, 'model_name')
            print('Model restored.')
            # algo.validate(sess)
