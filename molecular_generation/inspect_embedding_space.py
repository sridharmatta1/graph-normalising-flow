"""Diagnostic (not part of the phase pipeline): compares the EMPIRICAL
mean/variance of real training embeddings, after being pushed through
the trained N-conditioned flow's forward direction (f()), against the
prior network's own ASSUMED mu(N)/sigma(N).

inspect_conditional_prior.py only reports the latter -- what the model
is aiming for. This script reports the former -- what real data
actually produces once transformed -- which is the real test of
whether training converged to match its own target. The whole training
objective (log p(x) = log p(z) + log|det J|) is a bet that real data,
once transformed by f(), will actually look like the assumed prior; if
the empirical spread is much narrower or wider than sigma(N), that's
direct evidence of a mismatch between what training was aiming for and
what it actually achieved -- and a possible independent explanation
for the mode-concentration/diversity issue documented elsewhere in
this project (if real data occupies a narrower region than sigma(N)
assumes, sampling from the full assumed prior explores regions with
little real-data support).

Loads real per-atom embeddings straight from Phase 3's saved
embeddings_*.p chunks (the same files train_molecular_flow_n_conditioned.py
trained on), buckets them by molecule size N, and runs each bucket
through the SAME restored checkpoint inspect_conditional_prior.py uses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
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
flags.DEFINE_string('embeddings_dir', 'molecular_generation/data', '')
flags.DEFINE_integer('node_embedding_dim', 64, '')
flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_integer('flow_hidden_dim', 256, '')
flags.DEFINE_integer('flow_n_embed_dim', 32, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')
flags.DEFINE_string('n_values', '2,3,4,5,6,7,8,9',
                    'Comma-separated N values to check.')
flags.DEFINE_integer(
    'max_molecules_per_n', 300,
    'How many real molecules of each N to sample for the empirical '
    'estimate -- capped since some N are rare (e.g. N=2 has only 3 '
    'molecules in the entire training set).')
flags.DEFINE_string(
    'save_summary_json', '',
    'If non-empty, write the per-N empirical-vs-assumed comparison to '
    'this path as JSON, for turning into a chart locally.')
FLAGS = flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


def load_real_embeddings_by_n(embeddings_dir, n_values, max_per_n):
    """Returns {n: [num_molecules_of_size_n, n, node_embedding_dim]-ish
    list of per-molecule embedding arrays}, stopping early once every
    requested N has enough samples.
    """
    buckets = {n: [] for n in n_values}
    needed = set(n_values)
    files = sorted(glob.glob(os.path.join(embeddings_dir, 'embeddings_*.p')))
    if not files:
        raise FileNotFoundError(
            "No embeddings_*.p files found in {} -- run "
            "generate_molecular_embeddings.py first.".format(embeddings_dir))

    for path in files:
        with open(path, 'rb') as f:
            node_embeddings, n_node = pickle.load(f)
        cum = np.cumsum(n_node)
        start = 0
        for i, n in enumerate(n_node):
            end = cum[i]
            if n in needed and len(buckets[n]) < max_per_n:
                buckets[n].append(node_embeddings[start:end])
            start = end
        if all(len(buckets[n]) >= max_per_n for n in n_values):
            break
    return buckets


def main(argv):
    del argv
    n_values = [int(x) for x in FLAGS.n_values.split(',')]

    print("Loading real embeddings from {} ...".format(FLAGS.embeddings_dir))
    buckets = load_real_embeddings_by_n(FLAGS.embeddings_dir, n_values,
                                       FLAGS.max_molecules_per_n)
    for n in n_values:
        print("  N={}: {} real molecules found".format(n, len(buckets[n])))

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

        # Same construction order as inspect_conditional_prior.py /
        # generate_molecules_n_conditioned.py: f() first (real forward
        # pass -- this IS the quantity we want, not a throwaway dummy
        # call here), then the prior modules, so variable names line up
        # with the checkpoint.
        z, _log_det_jacobian = grevnet(graphs_tuple, inverse=True)
        n_embedding = prior_n_embedding_mod(graphs_tuple)
        mu, sigma = prior_net(n_embedding)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring N-conditioned flow from {}".format(FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)

        print("\n{:>4} {:>8} {:>16} {:>16} {:>16} {:>16}".format(
            "N", "n_mols", "empirical|z|", "empirical_std",
            "assumed_sigma", "assumed|mu|"))
        print("-" * 82)
        summary_rows = []
        for n in n_values:
            molecules = buckets[n]
            if not molecules:
                print("{:>4} {:>8}   (no real molecules of this size found)".format(
                    n, 0))
                continue
            node_embeddings = np.concatenate(molecules, axis=0)
            n_node_arr = np.array([n] * len(molecules), dtype=np.int32)
            feed = {
                node_embeddings_placeholder: node_embeddings,
                n_node_placeholder: n_node_arr,
            }
            z_val, mu_val, sigma_val = sess.run([z.nodes, mu, sigma],
                                                feed_dict=feed)
            empirical_abs_mean = float(np.mean(np.abs(z_val)))
            empirical_std = float(np.std(z_val))
            assumed_sigma_mean = float(np.mean(sigma_val))
            assumed_mu_abs_mean = float(np.mean(np.abs(mu_val)))
            print("{:>4} {:>8} {:>16.4f} {:>16.4f} {:>16.4f} {:>16.4f}".format(
                n, len(molecules), empirical_abs_mean, empirical_std,
                assumed_sigma_mean, assumed_mu_abs_mean))
            summary_rows.append({
                'n': n,
                'n_molecules_sampled': len(molecules),
                'empirical_abs_mean': empirical_abs_mean,
                'empirical_std': empirical_std,
                'assumed_sigma': assumed_sigma_mean,
                'assumed_mu_abs': assumed_mu_abs_mean,
            })

        if FLAGS.save_summary_json:
            with open(FLAGS.save_summary_json, 'w') as f:
                json.dump({'checkpoint': FLAGS.flow_checkpoint,
                          'by_n': summary_rows}, f, indent=2)
            print("\nSaved empirical-vs-assumed summary to {}".format(
                FLAGS.save_summary_json))


if __name__ == '__main__':
    app.run(main)
