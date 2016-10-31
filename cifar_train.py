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

d_feat = shared_template.construct(input=input_tensor)
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
                pred_labels_kmeans = np.array([], dtype=np.int16).reshape(0,)
                n_clusters = val_dataset.n_labels
                labels = []

                trainX = np.array([]).reshape(0, 0)

                def pool_features(feat, pool_type='avg'):
                    if pool_type == 'avg':
                        feat = feat.mean(axis=(1, 2))
                    if pool_type == 'max':
                        feat = feat.mean(axis=(1, 2))
                    return feat.reshape((feat.shape[0], feat.shape[-1]))

                print "Getting all the training features."
                for ii in range(val_dataset.batch_idx['train']):
                    x, _ = val_dataset.next_batch(batch_size, split='train')
                    d_features = sess.run(d_feat, {input_tensor: x})
                    d_features = pool_features(d_features, pool_type='avg')
                    if trainX.shape[0] == 0:  # Is empty
                        trainX = d_features
                    else:
                        trainX = np.concatenate((trainX, d_features), axis=0)
                print "Learning the clusters."
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(trainX)

                print "Extracting features from val set and predicting from it."
                for ii in range(val_dataset.batch_idx['val']):
                    x, batch_labels = val_dataset.next_batch(batch_size=batch_size, split="val")
                    feed_dict = {input_tensor: x}
                    d_out = sess.run([d_feat, d_prob], {input_tensor: x})
                    d_features = d_out[0]
                    pred_prob = d_out[1]
                    batch_pred_labels = np.argmax(pred_prob, axis=1)
                    pred_labels = np.concatenate((pred_labels, batch_pred_labels))
                    labels = labels + batch_labels

                    d_features = pool_features(d_features, pool_type='avg')
                    batch_pred_labels_kmeans = kmeans.predict(d_features)
                    pred_labels_kmeans = np.concatenate((pred_labels_kmeans, batch_pred_labels_kmeans))

                compute_cluster_scores(labels=np.asarray(labels), pred_labels=pred_labels, path=os.path.join('./scores.txt'))
                compute_cluster_scores(labels=np.asarray(labels), pred_labels=pred_labels_kmeans, path=os.path.join('./scores_kmeans.txt'))
