"""Diagnostic: do embeddings SAMPLED from a flow look like the REAL encoder
embeddings it was trained on? End to end, before the decoder.

inspect_latent_calibration.py showed the latent z of real data is
calibrated and Gaussian per dimension. That does not by itself guarantee
that sampling z from the prior and running the inverse flow reproduces the
real embedding distribution: it says nothing about correlations between
dimensions, or about whether the inverse really undoes the forward pass.
This script checks both, using the generators' exact sampling path:

  * round trip: g(f(x)) should equal x for real embeddings x (the
    architectural fix this project made; verified here on the molecular
    checkpoints).
  * sampled vs real, per molecule size N: per-dimension spread and mean,
    the 64x64 correlation structure, between-molecule variation, and an
    RBF-MMD against its noise floor (see latent_stats.compare_embeddings).

Works for the N-conditioned flow (optionally with --sigma_scale) and for the
unconditioned flow (optionally with --sample_temperature), so widened
sampling can be examined directly.
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
from conditional_prior import (ConditionalPriorNetwork,
                               sample_conditional_prior)
from gnn import dm_self_attn_gnn, make_mlp_model
from n_conditioning import NEmbedding

from latent_stats import compare_embeddings
from molecular_flow import MolecularGNFBlock, NConditionedMolecularGNFBlock
from tf_helpers import reset_sess, senders_receivers

flags.DEFINE_enum('model_kind', 'n_conditioned',
                  ['n_conditioned', 'unconditioned'], '')
flags.DEFINE_string('flow_checkpoint', '', 'Checkpoint prefix to restore.')
flags.DEFINE_string('embeddings_dir', 'molecular_generation/data',
                    'Real (training) embeddings_*.p files to compare against.')
flags.DEFINE_string('n_values', '7,8,9', 'Molecule sizes to compare.')
flags.DEFINE_integer('max_real_molecules', 2000, 'Per N.')
flags.DEFINE_integer('num_generated', 2000, 'Molecules sampled per N.')
flags.DEFINE_integer('sample_chunk', 250, 'Molecules per sampling call.')
flags.DEFINE_integer('roundtrip_molecules', 256, 'Per N.')
flags.DEFINE_float('sigma_scale', 1.0,
                   'N-conditioned only: multiplies sigma(N) when sampling.')
flags.DEFINE_float('sample_temperature', 1.0,
                   'Unconditioned only: multiplies the N(0,I) noise.')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_string('save_summary_json', '', '')
flags.DEFINE_integer('node_embedding_dim', 64, '')
flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')
flags.DEFINE_integer('flow_hidden_dim', 256, 'N-conditioned architecture.')
flags.DEFINE_integer('flow_n_embed_dim', 32, 'N-conditioned architecture.')
flags.DEFINE_integer('flow_latent_dim', 512, 'Unconditioned architecture.')
flags.DEFINE_integer('flow_num_layers', 3, 'Unconditioned architecture.')
flags.DEFINE_integer('flow_attn_kq_dim', 64, 'Unconditioned architecture.')
flags.DEFINE_integer('flow_attn_v_dim', 64, 'Unconditioned architecture.')
flags.DEFINE_integer('flow_attn_num_heads', 2, 'Unconditioned architecture.')
flags.DEFINE_integer('flow_attn_concat_heads_output_dim', 64,
                     'Unconditioned architecture.')
FLAGS = flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


def make_graphs_tuple(nodes, n_node):
    edges, globals_, receivers, senders, n_edge = transform_example(n_node)
    return gn.graphs.GraphsTuple(nodes=nodes, edges=edges, globals=globals_,
                                 receivers=receivers, senders=senders,
                                 n_node=n_node, n_edge=n_edge)


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
    """Returns (sess, node_ph, n_node_ph, recon_tensor, sample_n_ph,
    generated_tensor)."""
    dim = FLAGS.node_embedding_dim
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        node_ph = tf.placeholder(tf.float32, shape=[None, dim],
                                 name='node_embeddings_placeholder')
        n_node_ph = tf.placeholder(tf.int32, shape=[None],
                                   name='n_node_placeholder')
        sample_n_ph = tf.placeholder(tf.int32, shape=[None],
                                     name='sample_n_node_placeholder')
        graphs_tuple = make_graphs_tuple(node_ph, n_node_ph)
        sample_graph = make_graphs_tuple(
            tf.zeros([tf.reduce_sum(sample_n_ph), dim]), sample_n_ph)
        half_dim = dim // 2

        if FLAGS.model_kind == 'n_conditioned':
            grevnet = NConditionedMolecularGNFBlock(
                num_timesteps=FLAGS.flow_num_coupling_layers,
                node_embedding_dim=half_dim,
                hidden_dim=FLAGS.flow_hidden_dim,
                n_embed_dim=FLAGS.flow_n_embed_dim,
                weight_sharing=FLAGS.flow_weight_sharing,
                max_log_scale=FLAGS.flow_max_log_scale)
            prior_embedding = NEmbedding(FLAGS.flow_n_embed_dim)
            prior_net = ConditionalPriorNetwork(dim)
            # Same variable-creation order as training.
            z_real, _ = grevnet(graphs_tuple, inverse=True)
            prior_net(prior_embedding(graphs_tuple))
            recon = grevnet(z_real, inverse=False)

            mu, sigma = prior_net(prior_embedding(sample_graph))
            sampled = sample_conditional_prior(
                sample_n_ph, mu, sigma * FLAGS.sigma_scale)
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
            z_real, _ = grevnet(graphs_tuple, inverse=True)
            recon = grevnet(z_real, inverse=False)
            sampled = FLAGS.sample_temperature * tf.random_normal(
                [tf.reduce_sum(sample_n_ph), dim])

        generated = grevnet(sample_graph.replace(nodes=sampled),
                            inverse=False)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring {} flow from {}".format(FLAGS.model_kind,
                                                 FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)
    return sess, node_ph, n_node_ph, recon.nodes, sample_n_ph, generated.nodes


def main(argv):
    del argv
    if not FLAGS.flow_checkpoint:
        raise ValueError("--flow_checkpoint is required.")
    n_values = [int(x) for x in FLAGS.n_values.split(',')]
    dim = FLAGS.node_embedding_dim
    rng = np.random.RandomState(FLAGS.random_seed)

    print("Loading real embeddings from {} ...".format(FLAGS.embeddings_dir))
    buckets = load_by_n(FLAGS.embeddings_dir, n_values,
                        FLAGS.max_real_molecules)
    sess, node_ph, n_node_ph, recon_t, sample_n_ph, gen_t = (
        build_graph_and_restore())

    knob = ("sigma_scale={}".format(FLAGS.sigma_scale)
            if FLAGS.model_kind == 'n_conditioned'
            else "temperature={}".format(FLAGS.sample_temperature))
    print("\n" + "=" * 132)
    print("{} flow, {} | real = {}".format(FLAGS.model_kind, knob,
                                           FLAGS.embeddings_dir))
    print("=" * 132)
    print("{:>3} {:>5} {:>5} | {:>7} {:>13} {:>6} | {:>8} {:>8} {:>8} | "
          "{:>7} | {:>9} {:>9} | {:>9}".format(
              "N", "realM", "genM", "sd_rat", "p10-p90", "shift",
              "corr_rl", "corr_gn", "mismtch", "molmean", "mmd2", "null",
              "roundtrip"))
    print("-" * 132)
    results = {}
    for n in n_values:
        molecules = buckets[n]
        if len(molecules) < 20:
            print("{:>3} {:>5}   (too few real molecules)".format(
                n, len(molecules)))
            continue
        real_mols = np.stack(molecules).astype(np.float64)

        rel_errs, max_err = [], 0.0
        for lo in range(0, min(len(molecules), FLAGS.roundtrip_molecules),
                        128):
            chunk = molecules[lo:lo + 128]
            x = np.concatenate(chunk, axis=0)
            recon = sess.run(recon_t, feed_dict={
                node_ph: x,
                n_node_ph: np.array([n] * len(chunk), dtype=np.int32)})
            rel_errs.append(np.linalg.norm(recon - x) / np.linalg.norm(x))
            max_err = max(max_err, float(np.abs(recon - x).max()))
        roundtrip = float(np.mean(rel_errs))

        gen_chunks, produced = [], 0
        while produced < FLAGS.num_generated:
            k = min(FLAGS.sample_chunk, FLAGS.num_generated - produced)
            g = sess.run(gen_t, feed_dict={
                sample_n_ph: np.array([n] * k, dtype=np.int32)})
            gen_chunks.append(g.reshape(k, n, dim))
            produced += k
        gen_mols = np.concatenate(gen_chunks, axis=0)

        s = compare_embeddings(real_mols, gen_mols, rng)
        s['roundtrip_rel_err'] = roundtrip
        s['roundtrip_max_abs_err'] = max_err
        s['n_real_molecules'] = len(molecules)
        s['n_generated_molecules'] = len(gen_mols)
        results[n] = s
        print("{:>3} {:>5} {:>5} | {:>7.3f} {:>6.2f}-{:<6.2f} {:>6.3f} | "
              "{:>8.3f} {:>8.3f} {:>8.3f} | {:>7.3f} | {:>9.5f} {:>9.5f} | "
              "{:>9.2e}".format(
                  n, len(molecules), len(gen_mols), s['std_ratio_mean'],
                  s['std_ratio_p10'], s['std_ratio_p90'],
                  s['mean_shift_mean'], s['corr_offdiag_real'],
                  s['corr_offdiag_gen'], s['corr_mismatch_rel'],
                  s['molmean_std_ratio'], s['mmd2'], s['mmd2_null'],
                  roundtrip))

    print("\nsd_rat = per-dim std(generated)/std(real) (1.00 = same spread; "
          "p10-p90 across the {} dims); shift = mean |mean_gen-mean_real| / "
          "std_real; corr_rl/corr_gn = mean |off-diagonal correlation| of "
          "real/generated; mismtch = ||C_gen-C_real||_F relative to "
          "||C_real-I||_F (0 = same correlation structure); molmean = "
          "gen/real std of the per-molecule mean embedding; mmd2 vs null = "
          "RBF-MMD^2 real-vs-generated against real-vs-real noise floor; "
          "roundtrip = ||g(f(x))-x|| / ||x||.".format(dim))

    if FLAGS.save_summary_json:
        os.makedirs(os.path.dirname(FLAGS.save_summary_json) or '.',
                    exist_ok=True)
        with open(FLAGS.save_summary_json, 'w') as f:
            json.dump({'model_kind': FLAGS.model_kind,
                       'checkpoint': FLAGS.flow_checkpoint,
                       'sigma_scale': FLAGS.sigma_scale,
                       'sample_temperature': FLAGS.sample_temperature,
                       'results': {str(k): v for k, v in results.items()}},
                      f, indent=2)
        print("Saved summary to {}".format(FLAGS.save_summary_json))


if __name__ == '__main__':
    app.run(main)
