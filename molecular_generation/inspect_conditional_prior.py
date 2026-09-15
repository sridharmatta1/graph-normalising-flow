"""Diagnostic (not part of the phase pipeline): prints the trained
N-conditioned flow's learned (mu(N), sigma(N)) for a range of N values,
straight from the Phase 4 checkpoint.

Motivation: generate_molecules_n_conditioned.py's --target_n comparison
showed N-conditioning made uniqueness at a rare size (N=5) WORSE, not
better (25.5% vs the unconditioned flow's already-poor 32.9%), while
barely touching a common size (N=9, 96.5% vs 98.7%). The suspected
cause: ConditionalPriorNetwork fits (mu, sigma) per N by maximum
likelihood, and N=5 has very few real training examples to fit
against -- with that little signal, the network may fit an
overconfident (too narrow) sigma(5) rather than a well-calibrated one,
which would mechanically produce LESS diverse samples at N=5 than the
old flat N(0, I) prior did (same direction as the temperature sweep's
finding: a narrower effective sampling spread trades uniqueness away).

This script tests that diagnosis directly by reading sigma(N) back out
of the trained network for several N values, instead of only inferring
it indirectly from downstream generation metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

from absl import app
from absl import flags
import graph_nets as gn
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from conditional_prior import ConditionalPriorNetwork
from n_conditioning import NEmbedding

from molecular_flow import NConditionedMolecularGNFBlock
from tf_helpers import reset_sess, senders_receivers

flags.DEFINE_string(
    'flow_checkpoint',
    'molecular_generation/test_runs/flow_n_conditioned/checkpoints-100001',
    '')
flags.DEFINE_string('data_dir', 'molecular_generation/data', '')
flags.DEFINE_integer('node_embedding_dim', 64, '')
flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_integer('flow_hidden_dim', 256, '')
flags.DEFINE_integer('flow_n_embed_dim', 32, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')
flags.DEFINE_string('n_values', '2,3,4,5,6,7,8,9',
                    'Comma-separated N values to query sigma(N)/mu(N) for.')
FLAGS = flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


def main(argv):
    del argv
    n_values = [int(x) for x in FLAGS.n_values.split(',')]

    with open(os.path.join(FLAGS.data_dir, 'qm9_train.p'), 'rb') as f:
        train_examples = pickle.load(f)
    train_n_node = np.array([e['n_node'] for e in train_examples])
    counts = {n: int(np.sum(train_n_node == n)) for n in n_values}

    graph = tf.Graph()
    with graph.as_default():
        node_embeddings_placeholder = tf.placeholder(
            tf.float32, shape=[None, FLAGS.node_embedding_dim],
            name='node_embeddings_placeholder')
        n_node_placeholder = tf.placeholder(tf.int32, shape=[None],
                                            name='n_node_placeholder')
        edges, globals_, receivers, senders, n_edge = transform_example(
            n_node_placeholder)
        graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                             edges=edges, globals=globals_,
                                             receivers=receivers,
                                             senders=senders,
                                             n_node=n_node_placeholder,
                                             n_edge=n_edge)

        half_dim = FLAGS.node_embedding_dim // 2
        grevnet = NConditionedMolecularGNFBlock(
            num_timesteps=FLAGS.flow_num_coupling_layers,
            node_embedding_dim=half_dim,
            hidden_dim=FLAGS.flow_hidden_dim,
            n_embed_dim=FLAGS.flow_n_embed_dim,
            weight_sharing=FLAGS.flow_weight_sharing,
            max_log_scale=FLAGS.flow_max_log_scale)
        prior_n_embedding_mod = NEmbedding(FLAGS.flow_n_embed_dim)
        prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)

        # Same variable-creation order as training/generation: dummy f()
        # first, then the prior modules.
        grevnet(graphs_tuple, inverse=True)
        n_embedding = prior_n_embedding_mod(graphs_tuple)
        mu, sigma = prior_net(n_embedding)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring N-conditioned flow from {}".format(FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)

        print("\n{:>4} {:>14} {:>16} {:>16}".format(
            "N", "train_count", "sigma(N) mean", "|mu(N)| mean"))
        print("-" * 55)
        for n in n_values:
            # query_n_node has one graph of size n; NEmbedding only reads
            # graph.n_node, so the rest of the GraphsTuple content is
            # irrelevant here.
            feed = {
                node_embeddings_placeholder: np.zeros(
                    (n, FLAGS.node_embedding_dim), dtype=np.float32),
                n_node_placeholder: np.array([n], dtype=np.int32),
            }
            mu_val, sigma_val = sess.run([mu, sigma], feed_dict=feed)
            print("{:>4} {:>14} {:>16.4f} {:>16.4f}".format(
                n, counts.get(n, 0), float(np.mean(sigma_val)),
                float(np.mean(np.abs(mu_val)))))


if __name__ == '__main__':
    app.run(main)
