"""The conditional prior p(H_T | N).

Three pieces, in order:

1. ConditionalPriorNetwork: takes the *same* N-embedding Person B's FiLM
   generators use (passed in, never recomputed here) and outputs mu(N) and
   sigma(N), each of dimension d. sigma is forced positive via softplus,
   floored at min_sigma, and its output head is zero-weight/bias-initialized
   so sigma starts at exactly min_sigma_init (default ~1.0) regardless of
   the (also randomly initialized) N-embedding -- not an arbitrary value
   that can land near zero and blow up conditional_prior_log_prob. Per-node
   prior: every node in a graph with N nodes draws an i.i.d. vector from
   this one N(mu(N), sigma(N)^2).

2. conditional_prior_log_prob: given H_T (the flow output, stored as
   [total_num_nodes, d] node features the way every graph in this codebase
   is represented, not a padded [B, N, d] tensor) and the graph's N, computes
   the Gaussian log-density of each node vector under N(mu(N), sigma(N)^2),
   summed over nodes (per graph) and features -> [B]. This is
   log p_prior(H_T | N), meant to be added to B's total_logdet to form the
   training loss.

3. NHistogramPrior: turns a list of training-set node counts into a
   categorical distribution over observed N values, so generation can draw
   N itself (or a caller can override N explicitly, e.g. for Person D's
   controllability metric).
"""

import math

import numpy as np
import sonnet as snt
import tensorflow as tf

from n_conditioning import repeat_rows

LOG_TWO_PI = math.log(2.0 * math.pi)


class ConditionalPriorNetwork(snt.AbstractModule):
    """Maps the shared N-embedding to per-graph (mu, sigma) of dimension d.

    Takes the N-embedding as an input to _build rather than building its own
    NEmbedding, so this network and every one of B's FiLM generators read
    from exactly one N-embedding per forward pass.
    """

    def __init__(self,
                 output_dim,
                 mlp_hidden_dim=64,
                 min_sigma=0.1,
                 min_sigma_init=1.0,
                 name="ConditionalPriorNetwork"):
        super(ConditionalPriorNetwork, self).__init__(name=name)
        self.output_dim = output_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.min_sigma = min_sigma
        # softplus(pre_sigma_bias) + min_sigma == min_sigma_init at init.
        self._pre_sigma_bias_init = math.log(
            math.exp(min_sigma_init - min_sigma) - 1.0)

    def _build(self, n_embedding):
        hidden = tf.nn.relu(snt.Linear(self.mlp_hidden_dim,
                                      name="hidden")(n_embedding))
        # Zero-initialized so mu/sigma start at a fixed, sane value (0 and
        # min_sigma_init) regardless of the N-embedding's own random init,
        # instead of an arbitrary value that can make sigma collapse toward
        # min_sigma and blow up conditional_prior_log_prob. Training can
        # still move away from these defaults as it learns.
        mu = snt.Linear(
            self.output_dim,
            initializers={
                'w': tf.zeros_initializer(),
                'b': tf.zeros_initializer()
            },
            name="mu_head")(hidden)
        pre_sigma = snt.Linear(
            self.output_dim,
            initializers={
                'w': tf.zeros_initializer(),
                'b': tf.constant_initializer(self._pre_sigma_bias_init)
            },
            name="sigma_head")(hidden)
        sigma = tf.nn.softplus(pre_sigma) + self.min_sigma
        return mu, sigma


def _node_to_graph_ids(n_node):
    """[num_graphs] node counts -> [total_num_nodes] graph index per node."""
    num_graphs = tf.shape(n_node)[0]
    graph_ids = tf.range(num_graphs)
    return repeat_rows(tf.expand_dims(graph_ids, axis=1), n_node)[:, 0]


def conditional_prior_log_prob(h_t, n_node, mu, sigma, per_node=False):
    """log p_prior(H_T | N), summed over features, per graph or per node.

    Args:
      h_t: [total_num_nodes, d] flow output node features.
      n_node: [num_graphs] int node counts (graph.n_node).
      mu, sigma: [num_graphs, d] from ConditionalPriorNetwork.
      per_node: if True, skip the final per-graph aggregation and return the
        [total_num_nodes] per-node log-density instead -- needed anywhere
        that slices log-probs by node-index range (e.g. generate_graphs.py,
        which was written against the original per-node mvn.log_prob(nodes)
        convention and would otherwise silently index past the end of a
        [num_graphs]-shaped array).

    Returns:
      [num_graphs] (or [total_num_nodes] if per_node) log-density of each
      graph's nodes (or each individual node) under its own N-conditioned
      Gaussian.
    """
    node_mu = repeat_rows(mu, n_node)
    node_sigma = repeat_rows(sigma, n_node)

    z = (h_t - node_mu) / node_sigma
    per_node_per_feat_log_prob = -0.5 * (z * z) - tf.log(node_sigma) - 0.5 * LOG_TWO_PI
    per_node_log_prob = tf.reduce_sum(per_node_per_feat_log_prob, axis=1)

    if per_node:
        return per_node_log_prob

    graph_ids = _node_to_graph_ids(n_node)
    num_graphs = tf.shape(n_node)[0]
    return tf.unsorted_segment_sum(per_node_log_prob, graph_ids, num_graphs)


def sample_conditional_prior(n_node, mu, sigma, seed=None):
    """Draws H_T ~ p_prior(. | N): one i.i.d. N(mu(N), sigma(N)^2) vector per
    node, broadcasting each graph's mu/sigma across its N nodes.

    Returns:
      [total_num_nodes, d] sampled node features.
    """
    node_mu = repeat_rows(mu, n_node)
    node_sigma = repeat_rows(sigma, n_node)
    noise = tf.random_normal(tf.shape(node_mu), seed=seed)
    return node_mu + node_sigma * noise


class NHistogramPrior(object):
    """p(N): a categorical distribution over the training set's observed
    node counts, built from a plain histogram (no learned parameters).
    """

    def __init__(self, train_n_node_list):
        if len(train_n_node_list) == 0:
            raise ValueError("train_n_node_list must be non-empty.")
        values, counts = np.unique(
            np.asarray(train_n_node_list, dtype=np.int32), return_counts=True)
        self.values = values
        self.probs = counts.astype(np.float64) / counts.sum()

    def sample_np(self, batch_size, random_state=None):
        """Pure-numpy sampler, for use outside the TF graph (e.g. choosing N
        before building the generation graph).
        """
        rng = random_state or np.random
        return rng.choice(self.values, size=batch_size, p=self.probs).astype(np.int32)

    def empirical_histogram(self):
        """Returns (values, probs) for comparing a sampled N distribution
        back against the training histogram (Person C's done-check).
        """
        return self.values, self.probs
