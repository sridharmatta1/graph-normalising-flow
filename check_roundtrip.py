"""Checklist Step 2: full round-trip on real data (encode -> forward flow
-> inverse flow), compared back to the original real embeddings (Step 1's
input). If the flow's f() and g() were plumbed correctly, g(f(x)) should
approximately equal x for real x -- not exactly (batch norm's f() uses
fresh per-batch statistics while g() uses the stored running average, so
some residual is expected), but a LARGE mismatch here means the
generation-time normalization doesn't correctly invert what the training
direction did -- a plumbing bug, not a training-quality problem, and per
the checklist "the cheapest possible fix."

This is different from compare_real_vs_generated_embeddings.py, which
compares real x against g(z) for FRESH z ~ prior (tests whether the prior
+ flow reproduce the real distribution). This script instead compares x
against g(f(x)) for the SAME real x (tests whether the flow's forward and
inverse passes are actually consistent with each other) -- isolating
plumbing/normalization bugs from prior-calibration issues.

Read-only: restores the checkpoint but never modifies it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import warnings

from absl import flags
import graph_nets as gn
import numpy as np
import tensorflow as tf

from grevnet import NConditionedGNFBlock
from utils import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

flags.DEFINE_string('checkpoint', '', 'Trained GNF checkpoint to restore.')
flags.DEFINE_string(
    'embeddings_file', '',
    'Pickled (node_embeddings, n_node) file of REAL embeddings, e.g. '
    'results/ego_conditioned/seed_1/node_embeddings/embeddings_1_0.p')
flags.DEFINE_integer('node_embedding_dim', 14, 'Must match training.')
flags.DEFINE_integer('num_coupling_layers', 12, 'Must match training.')
flags.DEFINE_integer('latent_dim', 2048, 'Must match training.')
flags.DEFINE_integer('n_embed_dim', 32, 'Must match training.')
flags.DEFINE_bool('weight_sharing', False, 'Must match training.')
flags.DEFINE_float('max_log_scale', 2.0, 'Must match training.')
flags.DEFINE_integer('num_graphs_to_check', 16,
                     'How many real graphs (from the embeddings file) to '
                     'round-trip.')
FLAGS = tf.app.flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


# ============================================================
# Load REAL embeddings and pick a batch of whole graphs.
# ============================================================
with open(FLAGS.embeddings_file, 'rb') as f:
    real_node_embeddings, real_n_node = pickle.load(f)
real_node_embeddings = np.array(real_node_embeddings)
real_n_node = np.array(real_n_node)

num_graphs = min(FLAGS.num_graphs_to_check, len(real_n_node))
batch_n_node = real_n_node[:num_graphs]
total_nodes = int(np.sum(batch_n_node))
batch_nodes = real_node_embeddings[:total_nodes]
print("Round-tripping {} real graphs, {} total nodes, dim {}".format(
    num_graphs, total_nodes, batch_nodes.shape[1]))

# ============================================================
# Rebuild the model (same construction order as training).
# ============================================================
half_dim = FLAGS.node_embedding_dim // 2

n_node_placeholder = tf.placeholder(tf.int32, shape=[num_graphs],
                                    name='n_node_placeholder')
edges, globals_, receivers, senders, n_edge = transform_example(
    n_node_placeholder)
nodes_placeholder = tf.placeholder(tf.float32,
                                   shape=[None, FLAGS.node_embedding_dim],
                                   name='nodes_placeholder')
graph_phs = gn.graphs.GraphsTuple(nodes=nodes_placeholder,
                                  edges=edges,
                                  globals=globals_,
                                  receivers=receivers,
                                  senders=senders,
                                  n_node=n_node_placeholder,
                                  n_edge=n_edge)

grevnet = NConditionedGNFBlock(
    num_timesteps=FLAGS.num_coupling_layers,
    node_embedding_dim=half_dim,
    hidden_dim=FLAGS.latent_dim,
    n_embed_dim=FLAGS.n_embed_dim,
    weight_sharing=FLAGS.weight_sharing,
    use_batch_norm=True,
    max_log_scale=FLAGS.max_log_scale)

z_graph, _ = grevnet(graph_phs, inverse=True)  # f(x): real data -> z
x_roundtrip = grevnet(z_graph, inverse=False).nodes  # g(f(x))

sess = reset_sess()
saver = tf.train.Saver()
print("Restoring from {}".format(FLAGS.checkpoint))
saver.restore(sess, FLAGS.checkpoint)

feed_dict = {
    n_node_placeholder: batch_n_node,
    nodes_placeholder: batch_nodes,
}
x_roundtrip_val = sess.run(x_roundtrip, feed_dict=feed_dict)

diff = batch_nodes - x_roundtrip_val
per_node_diff_norm = np.linalg.norm(diff, axis=1)
per_node_x_norm = np.linalg.norm(batch_nodes, axis=1)

print("=" * 70)
print("Original x:      mean L2 norm = {:.4f}".format(
    np.mean(per_node_x_norm)))
print("Round-tripped g(f(x)): mean L2 norm = {:.4f}".format(
    np.mean(np.linalg.norm(x_roundtrip_val, axis=1))))
print("Mean |x - g(f(x))| per node (L2):    {:.4f}".format(
    np.mean(per_node_diff_norm)))
print("Mean relative error (||x-g(f(x))|| / ||x||): {:.4f}".format(
    np.mean(per_node_diff_norm / (per_node_x_norm + 1e-8))))
print()
if np.mean(per_node_diff_norm / (per_node_x_norm + 1e-8)) > 0.5:
    print("--> LARGE mismatch: g(f(x)) does not reconstruct x. This is a "
         "plumbing/normalization bug in the generation path (e.g. batch "
         "norm's train-time vs. generate-time statistics diverging "
         "further than expected), not a training-quality issue.")
else:
    print("--> Small mismatch: the flow's forward/inverse plumbing is "
         "reasonably consistent. The problem is more likely in the "
         "prior/sampling side (checklist step 5) than in f()/g() plumbing.")
print("=" * 70)
