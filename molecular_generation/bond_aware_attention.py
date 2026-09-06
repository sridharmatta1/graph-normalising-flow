"""Bond-type-aware variant of gnn.py's DMSelfAttentionMLP/DMSelfAttention
(the attention actually used by dm_self_attn_gnn, which
train_molecular_autoencoder.py uses for its encoder).

DMSelfAttention already restricts attention to real graph edges (each
atom only attends to itself and its actually-bonded neighbors, via
graph_nets' broadcast_sender_nodes_to_edges/broadcast_receiver_nodes_to_edges
over attention_graph.senders/.receivers) -- that part is not a gap, it
was confirmed correct by reading DMSelfAttention._build (gnn.py:496-556)
line by line. What it does NOT do is use the edge's bond-TYPE value
(single/double/triple, carried in attention_graph.edges) for anything --
attention weights and messages are computed purely from node embeddings,
so a double bond and a single bond currently produce identical attention
behavior. This module adds a learned per-bond-type bias to both the
attention logits (how much to attend) and the attended values (what
gets passed through), so bond type can actually influence message
passing, not just connectivity.

Lives here instead of gnn.py because gnn.py is a shared file used by
the existing community/ego pipeline; DMSelfAttentionMLP/DMSelfAttention
hardcode their attention step inline with no injectable hook, so
reusing them and swapping only the attention math isn't possible
without duplicating _build anyway. This mirrors their structure
closely rather than subclassing, to minimize the chance of silently
deviating from the original's (proven-working) behavior anywhere
other than the intended bond-type bias.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graph_nets as gn
import sonnet as snt
import tensorflow as tf


class BondAwareSelfAttention(snt.AbstractModule):
    """Same attention computation as gnn.py's DMSelfAttention, plus a
    learned per-(bond_type, head) bias added to the attention logits and
    a learned per-(bond_type, head, v_dim) bias added to the attended
    values -- both gathered per edge from attention_graph.edges (the
    bond-type index qm9_graph_data.py already stores there).
    """

    def __init__(self, kq_dim_division, kq_dim, v_dim, num_heads,
                num_bond_types, name="bond_aware_self_attention"):
        super(BondAwareSelfAttention, self).__init__(name=name)
        self._normalizer = gn.modules._unsorted_segment_softmax
        self._kq_dim_division = kq_dim_division
        self._kq_dim = kq_dim
        with self._enter_variable_scope():
            self._logit_bias = tf.get_variable(
                "bond_type_logit_bias", shape=[num_bond_types, num_heads],
                initializer=tf.zeros_initializer())
            self._value_bias = tf.get_variable(
                "bond_type_value_bias",
                shape=[num_bond_types, num_heads, v_dim],
                initializer=tf.zeros_initializer())

    def _build(self, node_values, node_keys, node_queries, attention_graph):
        sender_keys = gn.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_keys))
        sender_values = gn.blocks.broadcast_sender_nodes_to_edges(
            attention_graph.replace(nodes=node_values))
        receiver_queries = gn.blocks.broadcast_receiver_nodes_to_edges(
            attention_graph.replace(nodes=node_queries))

        edge_type = tf.cast(tf.reshape(attention_graph.edges, [-1]),
                            tf.int32)  # [num_edges]
        logit_bias = tf.gather(self._logit_bias, edge_type)  # [E, heads]
        value_bias = tf.gather(self._value_bias, edge_type)  # [E, heads, v]

        attention_weights_logits = tf.reduce_sum(sender_keys *
                                                 receiver_queries,
                                                 axis=-1)
        if self._kq_dim_division:
            attention_weights_logits /= tf.sqrt(
                tf.cast(self._kq_dim, dtype=tf.float32))
        # Bias is a freely-learned scale, added after the QK-dot-product's
        # own sqrt(dim) normalization so it isn't inadvertently shrunk by
        # a scaling factor that has nothing to do with it.
        attention_weights_logits += logit_bias

        normalized_attention_weights = gn.modules._received_edges_normalizer(
            attention_graph.replace(edges=attention_weights_logits),
            normalizer=self._normalizer)

        sender_values = sender_values + value_bias
        attended_edges = sender_values * normalized_attention_weights[..., None]

        received_edges_aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
            reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(
            attention_graph.replace(edges=attended_edges))

        return attention_graph.replace(nodes=aggregated_attended_values)


class BondAwareSelfAttentionMLP(snt.AbstractModule):
    """Same wrapper as gnn.py's DMSelfAttentionMLP (Q/K/V projections,
    output MLP, optional residual/layer-norm), using BondAwareSelfAttention
    in place of DMSelfAttention. Faithfully mirrors DMSelfAttentionMLP's
    exact projection/reshape/call pattern, including the (harmless,
    arbitrary) project_q/project_k argument-order quirk in the original's
    attn_module(...) call, to avoid introducing any behavioral difference
    other than the intended bond-type bias.
    """

    def __init__(self,
                kq_dim,
                v_dim,
                make_mlp_fn,
                num_bond_types,
                num_heads=8,
                concat_heads_output_dim=20,
                concat=True,
                residual=False,
                layer_norm=False,
                kq_dim_division=False,
                name="bond_aware_self_attention_mlp"):
        super(BondAwareSelfAttentionMLP, self).__init__(name=name)
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.mlp = make_mlp_fn()
        self.num_heads = num_heads
        self.num_bond_types = num_bond_types
        self.concat_heads_output_dim = concat_heads_output_dim
        self.concat = concat
        self.residual = residual
        self.layer_norm = layer_norm
        self.kq_dim_division = kq_dim_division

    def _build(self, graph):
        initializers = {
            'w': tf.contrib.layers.xavier_initializer(uniform=True),
        }

        project_q_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_q = project_q_mod(graph.nodes)
        project_k_mod = snt.Linear(self.num_heads * self.kq_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_k = project_k_mod(graph.nodes)

        project_q = tf.reshape(project_q, [-1, self.num_heads, self.kq_dim])
        project_k = tf.reshape(project_k, [-1, self.num_heads, self.kq_dim])

        project_v_mod = snt.Linear(self.v_dim,
                                   use_bias=False,
                                   initializers=initializers)
        project_v = project_v_mod(graph.nodes)
        project_v = tf.keras.backend.repeat(project_v, self.num_heads)

        attn_module = BondAwareSelfAttention(self.kq_dim_division,
                                             self.kq_dim, self.v_dim,
                                             self.num_heads,
                                             self.num_bond_types)
        attn_graph = attn_module(project_v, project_q, project_k, graph)

        new_nodes = attn_graph.nodes
        new_nodes = tf.reshape(new_nodes, [-1, self.num_heads * self.v_dim])

        new_node_proj = snt.Linear(self.concat_heads_output_dim,
                                   use_bias=False)
        new_nodes = new_node_proj(new_nodes)

        if self.concat:
            new_nodes = tf.concat([graph.nodes, new_nodes], axis=1)
        new_nodes = self.mlp(new_nodes)

        if self.residual:
            new_nodes += graph.nodes

        if self.layer_norm:
            ln_mod = snt.LayerNorm()
            new_nodes = ln_mod(new_nodes)
        return graph.replace(nodes=new_nodes)


def bond_aware_self_attn_gnn(kq_dim,
                             v_dim,
                             make_mlp_fn,
                             num_heads,
                             concat_heads_output_dim,
                             num_bond_types,
                             concat=True,
                             residual=False,
                             layer_norm=False,
                             kq_dim_division=False):
    return BondAwareSelfAttentionMLP(
        kq_dim=kq_dim,
        v_dim=v_dim,
        make_mlp_fn=make_mlp_fn,
        num_bond_types=num_bond_types,
        num_heads=num_heads,
        concat_heads_output_dim=concat_heads_output_dim,
        concat=concat,
        residual=residual,
        layer_norm=layer_norm,
        kq_dim_division=kq_dim_division)
