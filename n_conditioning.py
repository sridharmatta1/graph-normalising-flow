"""Injection point 1: condition the coupling-layer s/t networks on N.

N (the number of nodes in a graph) is turned into a per-graph embedding once
per forward pass, then fed into a small per-layer FiLM generator that
produces a scale/shift pair. That pair modulates the *hidden* features
inside the s/t network, before message passing runs.

Invariant this module exists to protect: N only ever touches the
computation of s and t. It never reaches the half of the nodes that passes
through unchanged, and it never appears in the log-det term directly (the
log-det stays sum(s) — N can only affect that sum through s's value, never
by entering the Jacobian as its own term).
"""

import graph_nets as gn
import sonnet as snt
import tensorflow as tf

from gnn import EDGE_BLOCK_OPT, IdentityModule


class NEmbedding(snt.AbstractModule):
    """Turns the per-graph integer node count into a [num_graphs, embed_dim]
    vector. Built once per forward pass and reused (read-only) by every
    coupling layer's FiLM generator.
    """

    def __init__(self, embed_dim, hidden_dim=64, name="NEmbedding"):
        super(NEmbedding, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def _build(self, graph):
        # graph.n_node has shape [num_graphs] and holds the node count of
        # each graph in the batch. log1p keeps the input on a reasonable
        # scale across very different N.
        n_node = tf.cast(tf.reshape(graph.n_node, [-1, 1]), tf.float32)
        feat = tf.log(n_node + 1.0)
        mlp = snt.nets.MLP(
            [self.hidden_dim, self.embed_dim],
            activation=tf.nn.relu,
            initializers={
                'w': tf.initializers.glorot_normal(),
                'b': tf.initializers.truncated_normal(stddev=0.1),
            },
            activate_final=False)
        return mlp(feat)


class FiLMGenerator(snt.AbstractModule):
    """Maps a shared N-embedding to this layer's own (gamma, beta).

    One instance per coupling layer/sub-network: they all read the same
    N-embedding but learn independent weights, so each layer decides for
    itself how strongly (and in what direction) N should modulate its
    hidden features.
    """

    def __init__(self, hidden_dim, mlp_hidden_dim=64, name="FiLMGenerator"):
        super(FiLMGenerator, self).__init__(name=name)
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim

    def _build(self, n_embedding):
        hidden = tf.nn.relu(
            snt.Linear(
                self.mlp_hidden_dim,
                initializers={
                    'w': tf.initializers.glorot_normal(),
                    'b': tf.initializers.truncated_normal(stddev=0.1),
                },
                name="hidden")(n_embedding))
        # Zero-initialized so (gamma, beta) start at exactly (1, 0) --
        # FiLM starts as a true no-op (feat*1+0 == feat) regardless of the
        # (also randomly initialized) N-embedding, instead of an arbitrary
        # random (gamma, beta) that perturbs feat by an uncontrolled amount
        # before any training has happened.
        out = snt.Linear(
            2 * self.hidden_dim,
            initializers={
                'w': tf.zeros_initializer(),
                'b': tf.zeros_initializer(),
            },
            name="gamma_beta")(hidden)
        gamma, beta = tf.split(out, num_or_size_splits=2, axis=1)
        # Centered at 1 so an untrained FiLM generator starts near identity.
        gamma = 1.0 + gamma
        return gamma, beta


def repeat_rows(tensor, repeats):
    """Repeats each row i of `tensor` `repeats[i]` times along axis 0.

    Equivalent to tf.repeat(tensor, repeats, axis=0), reimplemented because
    TF1.14 (this project's pinned version) predates tf.repeat, which the
    installed graph_nets release relies on internally. Shared by every
    per-graph -> per-node broadcast in this project (FiLM gamma/beta here,
    and the conditional prior's mu/sigma in conditional_prior.py).
    """
    cum = tf.cumsum(repeats)
    total = cum[-1]
    positions = tf.range(total)
    segment_ids = tf.searchsorted(cum, positions, side="right")
    return tf.gather(tensor, segment_ids, axis=0)


def broadcast_film_to_nodes(gamma, beta, graph):
    """Broadcasts per-graph gamma/beta ([num_graphs, H]) out to per-node
    ([total_num_nodes, H]), using graph.n_node to know how many nodes each
    graph's gamma/beta should repeat across.
    """
    node_gamma = repeat_rows(gamma, graph.n_node)
    node_beta = repeat_rows(beta, graph.n_node)
    return node_gamma, node_beta


def apply_film(feat, gamma, beta, graph):
    """feat: [total_num_nodes, H] hidden features for one half of the split.
    Modulates feat with the per-graph (gamma, beta), broadcast across the N
    nodes of each graph.
    """
    node_gamma, node_beta = broadcast_film_to_nodes(gamma, beta, graph)
    return node_gamma * feat + node_beta


class FiLMConditionedNodeBlock(snt.AbstractModule):
    """Node-update block used inside an s or t network:

      1. project the unchanged half into hidden features
      2. FiLM-modulate those hidden features with N's (gamma, beta)
      3. message-pass (aggregate neighbor features)
      4. project down to the output (s or t)

    FiLM happens before message passing, so N shapes how neighbors get
    aggregated, not just a post-hoc rescaling of the final output.
    """

    def __init__(self,
                 hidden_dim,
                 output_dim,
                 film_generator,
                 agg_fn=tf.unsorted_segment_mean,
                 name="FiLMConditionedNodeBlock"):
        super(FiLMConditionedNodeBlock, self).__init__(name=name)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.film_generator = film_generator
        self._received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            agg_fn)

    def _build(self, graph, n_embedding):
        project_in = snt.Linear(
            self.hidden_dim,
            initializers={
                'w': tf.initializers.glorot_normal(),
                'b': tf.initializers.truncated_normal(stddev=0.1),
            },
            name="film_project_in")
        feat = project_in(graph.nodes)

        gamma, beta = self.film_generator(n_embedding)
        feat = apply_film(feat, gamma, beta, graph)

        film_graph = graph.replace(nodes=feat)
        agg = self._received_edges_aggregator(film_graph)
        combined = tf.concat([feat, agg], axis=1)

        hidden = tf.nn.relu(
            snt.Linear(
                self.hidden_dim,
                initializers={
                    'w': tf.initializers.glorot_normal(),
                    'b': tf.initializers.truncated_normal(stddev=0.1),
                },
                name="film_project_out_hidden")(combined))
        # Zero-initialized: this layer directly produces s (or t, depending
        # on which of grevnet.py's two FiLMConditionedGNN instances this is).
        # Starting it at 0 means exp(s)==1 and t==0 at init, so the coupling
        # transform x1*exp(s)+t starts as the identity map. Without this, an
        # arbitrary-scale s from a generic initializer gets exponentiated
        # and compounds across every stacked coupling layer -- this is what
        # produced the millions-then-nan loss even after batch norm and the
        # conditional-prior sigma fix were already in place.
        new_nodes = snt.Linear(
            self.output_dim,
            initializers={
                'w': tf.zeros_initializer(),
                'b': tf.zeros_initializer(),
            },
            name="film_project_out")(hidden)
        return graph.replace(nodes=new_nodes)


class FiLMConditionedGNN(snt.AbstractModule):
    """An s or t network: wraps FiLMConditionedNodeBlock with the edge block
    needed so message passing has sender-node features to aggregate.
    """

    def __init__(self,
                 hidden_dim,
                 output_dim,
                 film_generator,
                 edge_block_opt=EDGE_BLOCK_OPT,
                 name="FiLMConditionedGNN"):
        super(FiLMConditionedGNN, self).__init__(name=name)
        with self._enter_variable_scope():
            self._edge_block = gn.blocks.EdgeBlock(
                edge_model_fn=IdentityModule, **edge_block_opt)
            self._node_block = FiLMConditionedNodeBlock(
                hidden_dim, output_dim, film_generator)

    def _build(self, graph, n_embedding):
        return self._node_block(self._edge_block(graph), n_embedding)


def make_film_conditioned_gnn_fn(hidden_dim, output_dim, mlp_hidden_dim=64):
    """Factory: returns a callable that builds one FiLMConditionedGNN, each
    with its own FiLMGenerator (own gamma/beta weights), so every coupling
    layer learns its own N-modulation while sharing the upstream N-embedding.
    """

    def make_gnn_fn():
        film_generator = FiLMGenerator(hidden_dim, mlp_hidden_dim=mlp_hidden_dim)
        return FiLMConditionedGNN(hidden_dim, output_dim, film_generator)

    return make_gnn_fn
