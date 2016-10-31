import sys; sys.path.append('.')
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli

import tensorflow as tf
import os
import infogan.misc.datasets as datasets
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
import numpy as np
import prettytensor as pt
from infogan.misc.utils import compute_cluster_scores
import infogan.models.discriminative_networks as D

batch_size = 128
dataset = datasets.Dataset(name='cifar',
                                   batch_size=128,
                                   output_size=64)
val_dataset = datasets.Dataset(name='cifar',
                                   batch_size=128,
                                   output_size=64)

D_model = D.dcgan_net(image_shape=dataset.image_shape,is_reg=False,encoder_dim=None)
shared_template = D_model.shared_template
discriminator_template = shared_template.custom_fully_connected(10)

input_tensor = tf.placeholder(tf.float32, [batch_size] + [dataset.output_size, dataset.output_size, 3])
label_tensor = tf.placeholder(tf.int64, [batch_size])

d_features = shared_template.construct(input=input_tensor)
d_logits = discriminator_template.construct(input=input_tensor)
d_prob = tf.nn.sigmoid(d_logits)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      d_logits, label_tensor, name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.scalar_summary("cross_entropy", cross_entropy_mean)
all_vars = tf.trainable_variables()
d_vars = [var for var in all_vars if var.name.startswith('d_')]
discriminator_optimizer = tf.train.AdamOptimizer(2e-3, beta1=0.5)
discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[cross_entropy_mean], var_list=d_vars)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('./logs/supervised/cifar', sess.graph)

    counter = 0
    for epoch in range(10):
        for i in range(dataset.batch_idx['train']):
            print("Epoch: %d Batch: %d/%d" % (epoch, i, dataset.batch_idx['train']))
            x, labels = dataset.next_batch(128)
            labels = np.asarray(labels)
            feed_dict = {input_tensor: x, label_tensor: labels}
            sess.run(discriminator_trainer, feed_dict)

            counter += 1

            summary_str = sess.run(summary_op, {input_tensor: x, label_tensor: labels})
            summary_writer.add_summary(summary_str, counter)

            if counter % 500 == 0:
                print "validating..."
                pred_labels = np.array([], dtype=np.int16).reshape(0,)
                labels = []
                for ii in range(val_dataset.batch_idx['val']):
                    x, batch_labels = val_dataset.next_batch(batch_size=batch_size, split="val")
                    feed_dict = {input_tensor: x}
                    pred_prob = sess.run(d_prob, {input_tensor: x})
                    batch_pred_labels = np.argmax(pred_prob, axis=1)
                    pred_labels = np.concatenate((pred_labels, batch_pred_labels))
                    labels = labels + batch_labels

                compute_cluster_scores(labels=np.asarray(labels), pred_labels=pred_labels, path=os.path.join('./scores.txt'))
