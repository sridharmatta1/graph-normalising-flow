"""Alternative coupling architectures for the N-conditioned molecular flow.

The original conditioned block's s/t network (n_conditioning.py's
FiLMConditionedNodeBlock) is shallow: one projection, FiLM, then the MEAN of
the raw neighbour inputs (on a fully connected graph that is one molecule-wide
average, identical for every atom) and a single hidden layer. It has no
atom-to-atom interaction at all. The variants below change one thing at a
time, all keeping FiLM conditioning on N and the zero-initialised output
layer (so every coupling layer starts as the identity):

  attn_film         (A) the unconditioned flow's self-attention GNN and its
                    3-layer width-512 MLP, with FiLM applied to the
                    attention's input features and to each MLP hidden layer.
  deep_film         (B) deeper/wider FiLM block: width 512, 3 rounds of
                    message passing over the hidden features, residual
                    connections, FiLM before every round.
  interaction_film  (C) pairwise interaction network: per-edge messages
                    computed from (sender, receiver) hidden features instead
                    of a mean, then aggregated, 2 rounds, residual.
  film + channel_permutation
                    (D) the original FiLM block with a fixed random channel
                    permutation between coupling layers (see
                    NConditionedMolecularGNFBlock).
  spline_film       (E) the original FiLM s/t network predicting
                    rational-quadratic-spline parameters instead of an
                    affine scale/shift (NConditionedSplineBlock).

build_n_conditioned_block() is the single place the trainer and the
generators construct the flow, so both always build identical graphs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from absl import flags
import graph_nets as gn
import sonnet as snt
import tensorflow as tf

from gnn import (EDGE_BLOCK_OPT, IdentityModule, dm_self_attn_gnn, get_gnns,
                 make_act_norm)
from n_conditioning import (FiLMGenerator, NEmbedding, apply_film,
                            broadcast_film_to_nodes,
                            make_film_conditioned_gnn_fn)

from molecular_flow import NConditionedMolecularGNFBlock
from spline_ops import TFOps, rqs_forward, rqs_inverse

COUPLING_ARCHS = ['film', 'attn_film', 'deep_film', 'interaction_film',
                  'spline_film']

# Size of the attn_film (A) s/t network. Defaults reproduce variant A exactly.
# Defined here, not in each script, so the trainer, the generator and the
# diagnostics all take the same flags and cannot drift apart; they must match
# the values the checkpoint was trained with.
flags.DEFINE_integer('coupling_attn_kq_dim', 64, 'attn_film only.')
flags.DEFINE_integer('coupling_attn_v_dim', 64, 'attn_film only.')
flags.DEFINE_integer('coupling_attn_num_heads', 2, 'attn_film only.')
flags.DEFINE_integer('coupling_attn_concat_dim', 64,
                     'attn_film only: concatenated-heads output width.')
flags.DEFINE_integer('coupling_attn_mlp_width', 512, 'attn_film only.')
flags.DEFINE_integer('coupling_attn_mlp_layers', 3, 'attn_film only.')
_FLAGS = flags.FLAGS


def attention_kwargs_from_flags():
    """FiLMAttentionGNN size arguments from the coupling_attn_* flags."""
    return dict(kq_dim=_FLAGS.coupling_attn_kq_dim,
                v_dim=_FLAGS.coupling_attn_v_dim,
                num_heads=_FLAGS.coupling_attn_num_heads,
                concat_heads_output_dim=_FLAGS.coupling_attn_concat_dim,
                mlp_width=_FLAGS.coupling_attn_mlp_width,
                mlp_layers=_FLAGS.coupling_attn_mlp_layers)


def _init():
    return {
        'w': tf.initializers.glorot_normal(),
        'b': tf.initializers.truncated_normal(stddev=0.1),
    }


def _zero_init():
    return {'w': tf.zeros_initializer(), 'b': tf.zeros_initializer()}


def _film(generator, feat, graph, n_embedding):
    gamma, beta = generator(n_embedding)
    return apply_film(feat, gamma, beta, graph)


# --------------------------------------------------------------------- A --
class _FiLMConditionedMLP(snt.AbstractModule):
    """MLP whose hidden activations are FiLM-modulated. dm_self_attn_gnn
    calls its MLP with the node features only, so the per-node (gamma, beta)
    pairs are handed in through set_film() just before the attention block is
    called."""

    def __init__(self, hidden_dim, num_layers, output_dim,
                 name="film_conditioned_mlp"):
        super(_FiLMConditionedMLP, self).__init__(name=name)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_dim = output_dim
        self._film = None

    def set_film(self, film):
        self._film = film

    def _build(self, x):
        for layer in range(self._num_layers - 1):
            x = tf.nn.relu(
                snt.Linear(self._hidden_dim, initializers=_init(),
                           name="hidden_%d" % layer)(x))
            gamma, beta = self._film[layer]
            x = gamma * x + beta
        return snt.Linear(self._output_dim, initializers=_zero_init(),
                          name="output")(x)


class FiLMAttentionGNN(snt.AbstractModule):

    def __init__(self, output_dim, input_dim, kq_dim=64, v_dim=64,
                 num_heads=2, concat_heads_output_dim=64, mlp_width=512,
                 mlp_layers=3, name="FiLMAttentionGNN"):
        super(FiLMAttentionGNN, self).__init__(name=name)
        with self._enter_variable_scope():
            self._film_input = FiLMGenerator(input_dim)
            self._film_hidden = [FiLMGenerator(mlp_width)
                                 for _ in range(mlp_layers - 1)]
            self._mlp = _FiLMConditionedMLP(mlp_width, mlp_layers,
                                            output_dim)
            self._attention = dm_self_attn_gnn(
                kq_dim=kq_dim,
                v_dim=v_dim,
                make_mlp_fn=lambda: self._mlp,
                num_heads=num_heads,
                concat_heads_output_dim=concat_heads_output_dim,
                concat=True,
                residual=False,
                layer_norm=False,
                kq_dim_division=True)

    def _build(self, graph, n_embedding):
        gamma, beta = self._film_input(n_embedding)
        node_gamma, node_beta = broadcast_film_to_nodes(gamma, beta, graph)
        graph = graph.replace(nodes=node_gamma * graph.nodes + node_beta)
        self._mlp.set_film([
            broadcast_film_to_nodes(*generator(n_embedding), graph)
            for generator in self._film_hidden
        ])
        return self._attention(graph)


# --------------------------------------------------------------------- B --
class FiLMDeepGNN(snt.AbstractModule):

    def __init__(self, output_dim, hidden_dim=512, num_rounds=3,
                 name="FiLMDeepGNN"):
        super(FiLMDeepGNN, self).__init__(name=name)
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._num_rounds = num_rounds
        with self._enter_variable_scope():
            self._edge_block = gn.blocks.EdgeBlock(
                edge_model_fn=IdentityModule, **EDGE_BLOCK_OPT)
            self._aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
                tf.unsorted_segment_mean)
            self._films = [FiLMGenerator(hidden_dim)
                           for _ in range(num_rounds + 1)]

    def _build(self, graph, n_embedding):
        feat = snt.Linear(self._hidden_dim, initializers=_init(),
                          name="project_in")(graph.nodes)
        for r in range(self._num_rounds):
            feat = _film(self._films[r], feat, graph, n_embedding)
            agg = self._aggregator(
                self._edge_block(graph.replace(nodes=feat)))
            update = tf.nn.relu(
                snt.Linear(self._hidden_dim, initializers=_init(),
                           name="update_%d" % r)(
                               tf.concat([feat, agg], axis=1)))
            feat = feat + update
        feat = _film(self._films[-1], feat, graph, n_embedding)
        hidden = tf.nn.relu(
            snt.Linear(self._hidden_dim, initializers=_init(),
                       name="project_out_hidden")(feat))
        new_nodes = snt.Linear(self._output_dim, initializers=_zero_init(),
                               name="project_out")(hidden)
        return graph.replace(nodes=new_nodes)


# --------------------------------------------------------------------- C --
class FiLMInteractionGNN(snt.AbstractModule):

    def __init__(self, output_dim, hidden_dim=256, num_rounds=2,
                 name="FiLMInteractionGNN"):
        super(FiLMInteractionGNN, self).__init__(name=name)
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._num_rounds = num_rounds
        with self._enter_variable_scope():
            self._edge_blocks = [
                gn.blocks.EdgeBlock(
                    edge_model_fn=lambda: snt.nets.MLP(
                        [hidden_dim, hidden_dim],
                        activation=tf.nn.relu,
                        activate_final=True,
                        initializers=_init(),
                        name="edge_mlp"),
                    use_edges=False,
                    use_receiver_nodes=True,
                    use_sender_nodes=True,
                    use_globals=False)
                for _ in range(num_rounds)
            ]
            self._aggregator = gn.blocks.ReceivedEdgesToNodesAggregator(
                tf.unsorted_segment_mean)
            self._films = [FiLMGenerator(hidden_dim)
                           for _ in range(num_rounds + 1)]

    def _build(self, graph, n_embedding):
        feat = snt.Linear(self._hidden_dim, initializers=_init(),
                          name="project_in")(graph.nodes)
        for r in range(self._num_rounds):
            feat = _film(self._films[r], feat, graph, n_embedding)
            messages = self._edge_blocks[r](graph.replace(nodes=feat))
            agg = self._aggregator(messages)
            update = tf.nn.relu(
                snt.Linear(self._hidden_dim, initializers=_init(),
                           name="update_%d" % r)(
                               tf.concat([feat, agg], axis=1)))
            feat = feat + update
        feat = _film(self._films[-1], feat, graph, n_embedding)
        hidden = tf.nn.relu(
            snt.Linear(self._hidden_dim, initializers=_init(),
                       name="project_out_hidden")(feat))
        new_nodes = snt.Linear(self._output_dim, initializers=_zero_init(),
                               name="project_out")(hidden)
        return graph.replace(nodes=new_nodes)


# --------------------------------------------------------------------- E --
class NConditionedSplineBlock(snt.AbstractModule):
    """Same structure as NConditionedMolecularGNFBlock (zero-init ActNorm on
    each half before it conditions the other half, FiLM conditioning on N),
    but each coupling sub-step transforms the other half with an elementwise
    monotonic rational-quadratic spline (spline_ops.py) whose parameters the
    FiLM s/t network predicts, instead of x * exp(s) + t. One network per
    sub-step (it outputs all spline parameters) instead of separate s and t
    networks. Zero raw parameters give the identity map, and the spline is
    bounded, so no max_log_scale clamp is needed.
    """

    def __init__(self,
                 num_timesteps,
                 node_embedding_dim,
                 hidden_dim,
                 n_embed_dim=32,
                 num_bins=8,
                 bound=5.0,
                 embedding_module=None,
                 name="NConditionedSplineBlock"):
        super(NConditionedSplineBlock, self).__init__(name=name)
        self.num_timesteps = num_timesteps
        self._half = node_embedding_dim
        self._num_bins = num_bins
        self._bound = bound
        params_per_dim = 3 * num_bins - 1
        with self._enter_variable_scope():
            self.n_embedding = (embedding_module if embedding_module is not None
                                else NEmbedding(n_embed_dim))
            make_net = make_film_conditioned_gnn_fn(
                hidden_dim, node_embedding_dim * params_per_dim)
            self.nets = [
                get_gnns(num_timesteps, make_net),
                get_gnns(num_timesteps, make_net),
            ]
            self.bns = [
                [make_act_norm(node_embedding_dim) for _ in range(num_timesteps)],
                [make_act_norm(node_embedding_dim) for _ in range(num_timesteps)],
            ]
            self._ops = TFOps()

    def _spline_params(self, raw_nodes):
        k = self._num_bins
        raw = tf.reshape(raw_nodes, [-1, self._half, 3 * k - 1])
        return raw[..., :k], raw[..., k:2 * k], raw[..., 2 * k:]

    def _transform(self, nodes, raw_nodes, inverse):
        raw_w, raw_h, raw_d = self._spline_params(raw_nodes)
        fn = rqs_inverse if inverse else rqs_forward
        out, log_deriv = fn(nodes, raw_w, raw_h, raw_d, self._ops,
                            num_bins=self._num_bins, bound=self._bound)
        return out, tf.reduce_sum(log_deriv)

    def f(self, x):
        log_det_jacobian = 0
        n_embedding = self.n_embedding(x)
        x0, x1 = tf.split(x.nodes, num_or_size_splits=2, axis=1)
        x0 = x.replace(nodes=x0)
        x1 = x.replace(nodes=x1)
        for i in range(self.num_timesteps):
            an = self.bns[0][i]
            log_det_jacobian += an.inverse_log_det_jacobian(x0.nodes)
            x0 = x0.replace(nodes=an.inverse(x0.nodes))
            raw = self.nets[0][i](x0, n_embedding).nodes
            new_x1, log_det = self._transform(x1.nodes, raw, inverse=False)
            log_det_jacobian += log_det
            x1 = x1.replace(nodes=new_x1)

            an = self.bns[1][i]
            log_det_jacobian += an.inverse_log_det_jacobian(x1.nodes)
            x1 = x1.replace(nodes=an.inverse(x1.nodes))
            raw = self.nets[1][i](x1, n_embedding).nodes
            new_x0, log_det = self._transform(x0.nodes, raw, inverse=False)
            log_det_jacobian += log_det
            x0 = x0.replace(nodes=new_x0)

        x = x.replace(nodes=tf.concat([x0.nodes, x1.nodes], axis=1))
        return x, log_det_jacobian

    def g(self, z):
        n_embedding = self.n_embedding(z)
        z0, z1 = tf.split(z.nodes, num_or_size_splits=2, axis=1)
        z0 = z.replace(nodes=z0)
        z1 = z.replace(nodes=z1)
        for i in reversed(range(self.num_timesteps)):
            raw = self.nets[1][i](z1, n_embedding).nodes
            new_z0, _ = self._transform(z0.nodes, raw, inverse=True)
            z1 = z1.replace(nodes=self.bns[1][i].forward(z1.nodes))
            z0 = z0.replace(nodes=new_z0)

            raw = self.nets[0][i](z0, n_embedding).nodes
            new_z1, _ = self._transform(z1.nodes, raw, inverse=True)
            z0 = z0.replace(nodes=self.bns[0][i].forward(z0.nodes))
            z1 = z1.replace(nodes=new_z1)
        return z.replace(nodes=tf.concat([z0.nodes, z1.nodes], axis=1))

    def _build(self, input, inverse=True):
        func = self.f if inverse else self.g
        return func(input)


# ---------------------------------------------------------------- factory --
def _s_t_factory(coupling_arch, hidden_dim, input_dim, attn_kwargs=None):
    """Returns factory(output_dim) -> make_gnn_fn, the shape
    NConditionedMolecularGNFBlock's make_gnn_fn_factory expects."""
    if coupling_arch == 'attn_film':
        def factory(output_dim):
            return lambda: FiLMAttentionGNN(output_dim, input_dim,
                                            **(attn_kwargs or {}))
    elif coupling_arch == 'deep_film':
        def factory(output_dim):
            return lambda: FiLMDeepGNN(output_dim)
    elif coupling_arch == 'interaction_film':
        def factory(output_dim):
            return lambda: FiLMInteractionGNN(output_dim)
    else:
        raise ValueError("No s/t factory for {}".format(coupling_arch))
    return factory


def build_n_conditioned_block(coupling_arch,
                              num_timesteps,
                              node_embedding_dim,
                              hidden_dim,
                              n_embed_dim,
                              weight_sharing,
                              max_log_scale,
                              embedding_module=None,
                              channel_permutation=False,
                              spline_num_bins=8,
                              spline_bound=5.0,
                              attn_kwargs=None):
    """node_embedding_dim is the FULL embedding width (64); the coupling
    halves are node_embedding_dim // 2. attn_kwargs sizes the attn_film
    network (attention_kwargs_from_flags()); None = variant A's defaults."""
    if coupling_arch not in COUPLING_ARCHS:
        raise ValueError("coupling_arch must be one of {}".format(
            COUPLING_ARCHS))
    half_dim = node_embedding_dim // 2
    if coupling_arch == 'spline_film':
        if weight_sharing or channel_permutation:
            raise ValueError("spline_film supports neither weight_sharing "
                             "nor channel_permutation.")
        return NConditionedSplineBlock(
            num_timesteps=num_timesteps,
            node_embedding_dim=half_dim,
            hidden_dim=hidden_dim,
            n_embed_dim=n_embed_dim,
            num_bins=spline_num_bins,
            bound=spline_bound,
            embedding_module=embedding_module)
    factory = (None if coupling_arch == 'film' else
               _s_t_factory(coupling_arch, hidden_dim, half_dim,
                            attn_kwargs))
    return NConditionedMolecularGNFBlock(
        num_timesteps=num_timesteps,
        node_embedding_dim=half_dim,
        hidden_dim=hidden_dim,
        n_embed_dim=n_embed_dim,
        weight_sharing=weight_sharing,
        max_log_scale=max_log_scale,
        embedding_module=embedding_module,
        make_gnn_fn_factory=factory,
        channel_permutation=channel_permutation)
