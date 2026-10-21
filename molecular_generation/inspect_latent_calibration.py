"""Diagnostic: is the flow's latent spread calibrated against its prior,
PER DIMENSION? Works for the N-conditioned flow (learned sigma(N)) and the
unconditioned flow (fixed N(0, I) prior), so the two can be compared.

Supersedes the comparison in inspect_embedding_space.py, which divided one
pooled std (over all atoms and all 64 dimensions) by the arithmetic mean of
the 64 per-dimension sigmas. That ratio is inflated whenever sigma differs
across dimensions, even for a perfectly calibrated prior. Here every
dimension is compared with its own sigma (see latent_stats.py), and tail
heaviness is measured too, since a Gaussian fit can match the variance and
still under-represent the tails.

Run once per model; each run restores one checkpoint and evaluates real
embeddings from each requested split (train and val).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
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
from gnn import dm_self_attn_gnn, make_mlp_model
from n_conditioning import NEmbedding

from latent_stats import (GAUSSIAN_ABS_RATIO, LatentCalibrationStats,
                          WithinMoleculeStats)
from molecular_flow import MolecularGNFBlock, NConditionedMolecularGNFBlock
from tf_helpers import reset_sess, senders_receivers

flags.DEFINE_enum('model_kind', 'n_conditioned',
                  ['n_conditioned', 'unconditioned'],
                  'Which flow the checkpoint is.')
flags.DEFINE_string('flow_checkpoint', '', 'Checkpoint prefix to restore.')
flags.DEFINE_string(
    'split_dirs',
    'train=molecular_generation/data,'
    'val=molecular_generation/data/val_embeddings',
    'Comma-separated name=directory pairs of embeddings_*.p files.')
flags.DEFINE_string('n_values', '4,5,6,7,8,9', '')
flags.DEFINE_integer('max_molecules_per_n', 2000, '')
flags.DEFINE_integer('chunk_molecules', 128, 'Molecules per forward pass.')
flags.DEFINE_string('save_summary_json', '', '')
flags.DEFINE_integer('node_embedding_dim', 64, '')
flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')
# N-conditioned flow architecture.
flags.DEFINE_integer('flow_hidden_dim', 256, '')
flags.DEFINE_integer('flow_n_embed_dim', 32, '')
# Unconditioned flow architecture.
flags.DEFINE_integer('flow_latent_dim', 512, '')
flags.DEFINE_integer('flow_num_layers', 3, '')
flags.DEFINE_integer('flow_attn_kq_dim', 64, '')
flags.DEFINE_integer('flow_attn_v_dim', 64, '')
flags.DEFINE_integer('flow_attn_num_heads', 2, '')
flags.DEFINE_integer('flow_attn_concat_heads_output_dim', 64, '')
FLAGS = flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


def load_by_n(embeddings_dir, n_values, max_per_n):
    buckets = {n: [] for n in n_values}
    files = sorted(glob.glob(os.path.join(embeddings_dir, 'embeddings_*.p')))
    if not files:
        raise FileNotFoundError(
            "No embeddings_*.p files in {}".format(embeddings_dir))
    for path in files:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        node_embeddings, n_node = d[0], d[1]
        cum = np.cumsum(n_node)
        start = 0
        for i, n in enumerate(n_node):
            end = cum[i]
            if n in buckets and len(buckets[n]) < max_per_n:
                buckets[n].append(node_embeddings[start:end])
            start = end
        if all(len(buckets[n]) >= max_per_n for n in n_values):
            break
    return buckets


def build_graph_and_restore():
    """Returns (sess, node_ph, n_node_ph, z_tensor, mu_tensor_or_None,
    sigma_tensor_or_None)."""
    graph = tf.Graph()
    with graph.as_default():
        node_ph = tf.placeholder(tf.float32,
                                 shape=[None, FLAGS.node_embedding_dim],
                                 name='node_embeddings_placeholder')
        n_node_ph = tf.placeholder(tf.int32, shape=[None],
                                   name='n_node_placeholder')
        edges, globals_, receivers, senders, n_edge = transform_example(
            n_node_ph)
        graphs_tuple = gn.graphs.GraphsTuple(
            nodes=node_ph, edges=edges, globals=globals_,
            receivers=receivers, senders=senders, n_node=n_node_ph,
            n_edge=n_edge)
        half_dim = FLAGS.node_embedding_dim // 2

        mu = sigma = None
        if FLAGS.model_kind == 'n_conditioned':
            grevnet = NConditionedMolecularGNFBlock(
                num_timesteps=FLAGS.flow_num_coupling_layers,
                node_embedding_dim=half_dim,
                hidden_dim=FLAGS.flow_hidden_dim,
                n_embed_dim=FLAGS.flow_n_embed_dim,
                weight_sharing=FLAGS.flow_weight_sharing,
                max_log_scale=FLAGS.flow_max_log_scale)
            prior_embedding = NEmbedding(FLAGS.flow_n_embed_dim)
            prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)
            z, _ = grevnet(graphs_tuple, inverse=True)
            mu, sigma = prior_net(prior_embedding(graphs_tuple))
        else:
            make_mlp_fn = partial(make_mlp_model, FLAGS.flow_latent_dim,
                                  half_dim, FLAGS.flow_num_layers)
            attn_gnn_fn = partial(
                dm_self_attn_gnn,
                kq_dim=FLAGS.flow_attn_kq_dim,
                v_dim=FLAGS.flow_attn_v_dim,
                make_mlp_fn=make_mlp_fn,
                num_heads=FLAGS.flow_attn_num_heads,
                concat_heads_output_dim=FLAGS.flow_attn_concat_heads_output_dim,
                concat=True,
                residual=False,
                layer_norm=False,
                kq_dim_division=True)
            grevnet = MolecularGNFBlock(attn_gnn_fn,
                                        FLAGS.flow_num_coupling_layers,
                                        half_dim, FLAGS.flow_weight_sharing,
                                        FLAGS.flow_max_log_scale)
            z, _ = grevnet(graphs_tuple, inverse=True)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring {} flow from {}".format(FLAGS.model_kind,
                                                 FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)
    return sess, node_ph, n_node_ph, z.nodes, mu, sigma


def main(argv):
    del argv
    if not FLAGS.flow_checkpoint:
        raise ValueError("--flow_checkpoint is required.")
    n_values = [int(x) for x in FLAGS.n_values.split(',')]
    split_dirs = dict(p.split('=', 1) for p in FLAGS.split_dirs.split(','))

    sess, node_ph, n_node_ph, z_t, mu_t, sigma_t = build_graph_and_restore()
    dim = FLAGS.node_embedding_dim
    results = {}

    for split, directory in split_dirs.items():
        print("\n" + "=" * 118)
        print("{} flow, {} embeddings ({})".format(FLAGS.model_kind, split,
                                                   directory))
        print("=" * 118)
        print("{:>3} {:>6} | {:>6} {:>6} {:>5} | {:>6} {:>13} {:>7} {:>7} | "
              "{:>6} {:>7} | {:>7}".format(
                  "N", "mols", "sig_mn", "sig_gm", "nfloor", "zRMS",
                  "dim p10-p90", "%wide", "%narrw", "kurt", "abs/rms",
                  "pool/sg"))
        print("-" * 118)
        buckets = load_by_n(directory, n_values, FLAGS.max_molecules_per_n)
        results[split] = {}
        for n in n_values:
            molecules = buckets[n]
            if not molecules:
                print("{:>3} {:>6}   (no molecules)".format(n, 0))
                continue
            stats = LatentCalibrationStats(dim)
            within = WithinMoleculeStats(dim)
            for lo in range(0, len(molecules), FLAGS.chunk_molecules):
                chunk = molecules[lo:lo + FLAGS.chunk_molecules]
                feed = {
                    node_ph: np.concatenate(chunk, axis=0),
                    n_node_ph: np.array([n] * len(chunk), dtype=np.int32),
                }
                if mu_t is not None:
                    z, mu, sigma = sess.run([z_t, mu_t, sigma_t],
                                            feed_dict=feed)
                    mu, sigma = mu[0], sigma[0]
                else:
                    z = sess.run(z_t, feed_dict=feed)
                    mu, sigma = np.zeros(dim), np.ones(dim)
                stats.update(z, mu, sigma)
                within.update(z.reshape(len(chunk), n, dim), mu, sigma)
            s = stats.summary()
            s['n_molecules'] = len(molecules)
            s['within_molecule'] = within.summary()
            results[split][n] = s
            print("{:>3} {:>6} | {:>6.3f} {:>6.3f} {:>5d} | {:>6.3f} "
                  "{:>6.2f}-{:<6.2f} {:>6.1f}% {:>6.1f}% | {:>6.2f} {:>7.3f} "
                  "| {:>7.3f}".format(
                      n, len(molecules), s['sigma_mean'], s['sigma_geomean'],
                      s['dims_sigma_near_floor'], s['zscore_rms'],
                      s['zscore_rms_dim_p10'], s['zscore_rms_dim_p90'],
                      100 * s['frac_dims_data_wider'],
                      100 * s['frac_dims_data_narrower'],
                      s['kurtosis_mean'], s['abs_ratio'],
                      s['pooled_over_sigma_mean']))

        print("\nWithin-molecule independence of the latent atoms ({} "
              "embeddings). DE = 1 means atoms of one molecule are "
              "independent, as the prior assumes; DE > 1 means they share "
              "a molecule-level component (ICC = (DE-1)/(N-1)). DE_prior is "
              "measured against the prior's own sigma.".format(split))
        print("{:>3} {:>6} | {:>7} {:>13} | {:>7} {:>13} | {:>7}".format(
            "N", "mols", "DE", "p10-p90", "DE_prior", "p10-p90", "ICC"))
        print("-" * 78)
        for n in n_values:
            if n not in results[split]:
                continue
            w = results[split][n]['within_molecule']
            print("{:>3} {:>6} | {:>7.3f} {:>6.2f}-{:<6.2f} | {:>7.3f} "
                  "{:>6.2f}-{:<6.2f} | {:>7.4f}".format(
                      n, results[split][n]['n_molecules'],
                      w['design_effect'], w['design_effect_p10'],
                      w['design_effect_p90'], w['design_effect_prior'],
                      w['design_effect_prior_p10'],
                      w['design_effect_prior_p90'], w['icc']))

    print("\nColumns: sig_mn/sig_gm = arithmetic/geometric mean of the "
          "prior's per-dim sigma; nfloor = dims with sigma < 0.15 "
          "(floor is 0.1); zRMS = sqrt(mean over dims of E[((z-mu)/sigma)^2])"
          " -- 1.00 means calibrated, 1.5 means data is 1.5x wider than the "
          "prior; dim p10-p90 = spread of that ratio across the 64 dims; "
          "%wide/%narrw = dims with ratio >1.25 / <0.8; kurt = raw "
          "kurtosis (Gaussian = 3); abs/rms = E|u|/sqrt(E[u^2]) (Gaussian = "
          "{:.3f}); pool/sg = the OLD pooled-std / mean-sigma statistic."
          .format(GAUSSIAN_ABS_RATIO))

    if FLAGS.save_summary_json:
        os.makedirs(os.path.dirname(FLAGS.save_summary_json) or '.',
                    exist_ok=True)
        with open(FLAGS.save_summary_json, 'w') as f:
            json.dump({'model_kind': FLAGS.model_kind,
                       'checkpoint': FLAGS.flow_checkpoint,
                       'results': results}, f, indent=2)
        print("Saved summary to {}".format(FLAGS.save_summary_json))


if __name__ == '__main__':
    app.run(main)
