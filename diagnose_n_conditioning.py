"""Checks whether N is actually engaged by a trained N-conditioned GNF
checkpoint, rather than the FiLM/prior pathways having settled into a
near-no-op regardless of N.

Rebuilds the same architecture as train_grevnet_conditioned_with_data.py
(same classes, same construction order, so variable names line up with the
checkpoint), restores its trained weights, then evaluates -- for several
different N values, all in one batch -- the two places N enters the model:

  1. The conditional prior's mu(N)/sigma(N) (conditional_prior.py).
  2. One coupling layer's FiLM gamma(N)/beta(N) (n_conditioning.py), via
     the flow's own internal N-embedding (a separate, independently-learned
     embedding from the prior's -- see run_grevnet_conditioned.py's docs).

Prints each quantity per N, plus its std-dev *across* N: if that std is
near zero relative to the quantity's own scale, N-conditioning has
collapsed to a no-op; a clearly nonzero std is evidence N is doing real
work in that pathway.

Node feature *values* fed in are irrelevant here (dummy zeros) -- every
quantity this script inspects (n_embedding, mu, sigma, gamma, beta) is a
function of graph.n_node only, never of graph.nodes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from absl import flags
import graph_nets as gn
import numpy as np
import tensorflow as tf

from conditional_prior import ConditionalPriorNetwork
from grevnet import NConditionedGNFBlock
from n_conditioning import NEmbedding
from utils import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

flags.DEFINE_string(
    'checkpoint', '',
    'Exact checkpoint path prefix to restore, e.g. '
    '.../checkpoints-501 (no .index/.data suffix). Lets you target a '
    'specific checkpoint directly instead of always restoring whichever '
    'one a directory\'s "checkpoint" metadata file marks as latest.')
flags.DEFINE_string('probe_n_values', '4,8,12,16,20',
                    'Comma-separated N values to probe, one per graph.')
flags.DEFINE_integer('node_embedding_dim', 14, 'Must match training.')
flags.DEFINE_integer('num_coupling_layers', 12, 'Must match training.')
flags.DEFINE_integer('latent_dim', 2048,
                     'Hidden dim used inside the FiLM-conditioned s/t '
                     'networks. Must match training.')
flags.DEFINE_integer('n_embed_dim', 32, 'Must match training.')
flags.DEFINE_bool('weight_sharing', True, 'Must match training.')
flags.DEFINE_float('max_log_scale', 2.0, 'Must match training.')
FLAGS = tf.app.flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


probe_n = [int(x) for x in FLAGS.probe_n_values.split(',')]
num_graphs = len(probe_n)
half_dim = FLAGS.node_embedding_dim // 2

n_node_placeholder = tf.placeholder(tf.int32, shape=[num_graphs],
                                    name='n_node_placeholder')
edges, globals_, receivers, senders, n_edge = transform_example(
    n_node_placeholder)
dummy_nodes = tf.zeros(
    [tf.reduce_sum(n_node_placeholder), FLAGS.node_embedding_dim])
graph_phs = gn.graphs.GraphsTuple(nodes=dummy_nodes,
                                  edges=edges,
                                  globals=globals_,
                                  receivers=receivers,
                                  senders=senders,
                                  n_node=n_node_placeholder,
                                  n_edge=n_edge)

# Same construction as train_grevnet_conditioned_with_data.py, so variable
# scopes/names match the checkpoint.
grevnet = NConditionedGNFBlock(
    num_timesteps=FLAGS.num_coupling_layers,
    node_embedding_dim=half_dim,
    hidden_dim=FLAGS.latent_dim,
    n_embed_dim=FLAGS.n_embed_dim,
    weight_sharing=FLAGS.weight_sharing,
    use_batch_norm=True,
    max_log_scale=FLAGS.max_log_scale)

prior_n_embedding_mod = NEmbedding(FLAGS.n_embed_dim)
prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)

# Forces every submodule's variables to be created in the same order/names
# as training (grevnet.n_embedding, every coupling layer's FiLMGenerator,
# every batch-norm bijector), so the Saver below finds matches.
_ = grevnet(graph_phs, inverse=True)

prior_n_embedding = prior_n_embedding_mod(graph_phs)
prior_mu, prior_sigma = prior_net(prior_n_embedding)

# grevnet's own N-embedding (separate weights from prior_n_embedding_mod --
# see run_grevnet_conditioned.py's docstring). Re-calling an already-built
# sonnet module is safe and just recomputes its (now-trained) output.
flow_n_embedding = grevnet.n_embedding(graph_phs)
film_generator = (grevnet.s[0]._node_block.film_generator
                  if FLAGS.weight_sharing else
                  grevnet.s[0][0]._node_block.film_generator)
gamma, beta = film_generator(flow_n_embedding)

sess = reset_sess()
if FLAGS.checkpoint:
    saver = tf.train.Saver()
    print("Restoring from {}".format(FLAGS.checkpoint))
    saver.restore(sess, FLAGS.checkpoint)
else:
    print("No --checkpoint given -- using random initialization "
          "(reset_sess() already ran global_variables_initializer()).")

values = sess.run(
    {
        'prior_mu': prior_mu,
        'prior_sigma': prior_sigma,
        'gamma': gamma,
        'beta': beta,
    }, feed_dict={n_node_placeholder: probe_n})


def report(name, arr):
    """arr: [num_graphs, dim]. Prints per-N summary stats, the std *across*
    N of each dim (relative to that quantity's own scale -- misleading to
    compare directly between quantities with a nonzero identity baseline
    like sigma/gamma (~1.0) vs. a zero baseline like mu/beta), and the raw
    (non-relative) absolute std across N, which *is* fair to compare
    across quantities regardless of baseline.
    """
    print("\n{} (shape {}):".format(name, arr.shape))
    for n, row in zip(probe_n, arr):
        print("  N={:>3}  mean={: .4f}  norm={: .4f}".format(
            n, np.mean(row), np.linalg.norm(row)))
    per_dim_std_across_n = np.std(arr, axis=0)
    per_dim_scale = np.mean(np.abs(arr), axis=0) + 1e-8
    relative_variation = np.mean(per_dim_std_across_n / per_dim_scale)
    absolute_variation = np.mean(per_dim_std_across_n)
    print("  --> mean relative std across N: {:.4f}  ({})".format(
        relative_variation, "N IS influencing this"
        if relative_variation > 0.05 else "N-conditioning looks collapsed"))
    print("  --> mean absolute std across N: {:.6f}  (fair to compare "
         "across quantities, unlike the relative number above)".format(
             absolute_variation))


print("=" * 70)
print("Probing N in {}".format(probe_n))
report("prior mu(N)", values['prior_mu'])
report("prior sigma(N)", values['prior_sigma'])
report("coupling-layer gamma(N)", values['gamma'])
report("coupling-layer beta(N)", values['beta'])
print("=" * 70)
