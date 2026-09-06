"""Unconditioned GNF coupling-layer block for molecular embeddings --
same structure as grevnet.py's GNFBlock (the plain, non-N-conditioned
flow used for the community/ego baseline), but with the two fixes
already found and proven necessary earlier in this project for
NConditionedGNFBlock:
  - ActNorm (gnn.py) instead of BatchNorm, for exact invertibility
    between the training direction (f) and generation direction (g) --
    BatchNorm's f()/.inverse() uses fresh per-batch statistics while
    g()/.forward() uses the stored running average, so they're not
    exact inverses of each other.
  - max_log_scale * tanh(...) clamping on the affine coupling scale s,
    to stop exp(s) compounding unboundedly across num_timesteps
    stacked coupling layers.

Needed here because train_molecular_flow.py's first real run (plain
GNFBlock, no clamping) diverged to NaN within ~1300 iterations --
log_det_jacobian grew without bound every single logged step right up
to the NaN, the exact signature this project saw and fixed for
NConditionedGNFBlock. GNFBlock was only ever validated stable against
community/ego's own encoder's embedding distribution; molecular
embeddings evidently have different enough scale characteristics that
the same instability applies to the unconditioned flow too.

Lives here instead of modifying grevnet.py's GNFBlock, which is shared
with the working, unmodified community/ego baseline. This is
NConditionedGNFBlock's exact fix, with the N-embedding/FiLM-conditioning
parts stripped out (N-conditioning is deliberately deferred, matching
the original project plan) -- plain make_gnn_fn-based s/t networks,
same as GNFBlock uses, not FiLM-conditioned ones.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import sonnet as snt
import tensorflow as tf

from gnn import get_gnns, make_act_norm


class MolecularGNFBlock(snt.AbstractModule):
    def __init__(self,
                make_gnn_fn,
                num_timesteps,
                node_embedding_dim,
                weight_sharing=False,
                max_log_scale=2.0,
                name="MolecularGNFBlock"):
        super(MolecularGNFBlock, self).__init__(name=name)
        self.num_timesteps = num_timesteps
        self.weight_sharing = weight_sharing
        self.max_log_scale = max_log_scale
        with self._enter_variable_scope():
            if weight_sharing:
                self.s = [make_gnn_fn(), make_gnn_fn()]
                self.t = [make_gnn_fn(), make_gnn_fn()]
            else:
                self.s = [
                    get_gnns(num_timesteps, make_gnn_fn),
                    get_gnns(num_timesteps, make_gnn_fn)
                ]
                self.t = [
                    get_gnns(num_timesteps, make_gnn_fn),
                    get_gnns(num_timesteps, make_gnn_fn)
                ]
            self.bns = [
                [make_act_norm(node_embedding_dim) for _ in range(num_timesteps)],
                [make_act_norm(node_embedding_dim) for _ in range(num_timesteps)],
            ]

    def f(self, x):
        log_det_jacobian = 0
        x0, x1 = tf.split(x.nodes, num_or_size_splits=2, axis=1)
        x0 = x.replace(nodes=x0)
        x1 = x.replace(nodes=x1)
        for i in range(self.num_timesteps):
            an = self.bns[0][i]
            log_det_jacobian += an.inverse_log_det_jacobian(x0.nodes)
            x0 = x0.replace(nodes=an.inverse(x0.nodes))
            if self.weight_sharing:
                s = self.s[0](x0).nodes
                t = self.t[0](x0).nodes
            else:
                s = self.s[0][i](x0).nodes
                t = self.t[0][i](x0).nodes
            s = self.max_log_scale * tf.tanh(s / self.max_log_scale)
            log_det_jacobian += tf.reduce_sum(s)
            x1 = x1.replace(nodes=x1.nodes * tf.exp(s) + t)

            an = self.bns[1][i]
            log_det_jacobian += an.inverse_log_det_jacobian(x1.nodes)
            x1 = x1.replace(nodes=an.inverse(x1.nodes))
            if self.weight_sharing:
                s = self.s[1](x1).nodes
                t = self.t[1](x1).nodes
            else:
                s = self.s[1][i](x1).nodes
                t = self.t[1][i](x1).nodes
            s = self.max_log_scale * tf.tanh(s / self.max_log_scale)
            log_det_jacobian += tf.reduce_sum(s)
            x0 = x0.replace(nodes=x0.nodes * tf.exp(s) + t)

        x = x.replace(nodes=tf.concat([x0.nodes, x1.nodes], axis=1))
        return x, log_det_jacobian

    def g(self, z):
        z0, z1 = tf.split(z.nodes, num_or_size_splits=2, axis=1)
        z0 = z.replace(nodes=z0)
        z1 = z.replace(nodes=z1)
        for i in reversed(range(self.num_timesteps)):
            if self.weight_sharing:
                s = self.s[1](z1).nodes
                t = self.t[1](z1).nodes
            else:
                s = self.s[1][i](z1).nodes
                t = self.t[1][i](z1).nodes
            s = self.max_log_scale * tf.tanh(s / self.max_log_scale)
            z1 = z1.replace(nodes=self.bns[1][i].forward(z1.nodes))
            z0 = z0.replace(nodes=(z0.nodes - t) * tf.exp(-s))

            if self.weight_sharing:
                s = self.s[0](z0).nodes
                t = self.t[0](z0).nodes
            else:
                s = self.s[0][i](z0).nodes
                t = self.t[0][i](z0).nodes
            s = self.max_log_scale * tf.tanh(s / self.max_log_scale)
            z0 = z0.replace(nodes=self.bns[0][i].forward(z0.nodes))
            z1 = z1.replace(nodes=(z1.nodes - t) * tf.exp(-s))
        return z.replace(nodes=tf.concat([z0.nodes, z1.nodes], axis=1))

    def _build(self, input, inverse=True):
        func = self.f if inverse else self.g
        return func(input)
