"""Phase 4 extension: joint N + chemical-property conditioning.

Mirrors n_conditioning.py's NEmbedding exactly (log(N+1) through a
small MLP), but concatenates a second, normalized feature -- a real
QM9 property (e.g. 'mu', dipole moment) -- before the MLP, so the
resulting embedding carries both signals. Everything downstream (FiLM
generators, ConditionalPriorNetwork) is agnostic to what produced its
input embedding, so this plugs into the exact same machinery
NConditionedMolecularGNFBlock already uses for N-only conditioning --
see that class's embedding_module parameter.

Lives in molecular_generation/, not n_conditioning.py (root, shared
with community/ego), since this is molecular-specific and the root
file's NEmbedding must stay untouched for the working community/ego
pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class NPropertyEmbedding(snt.AbstractModule):
    """Turns (N, a real-valued property) into a [num_graphs, embed_dim]
    vector -- the joint-conditioning analogue of n_conditioning.py's
    NEmbedding (which only takes N).

    The property is read from graph.globals -- qm9_graph_data.py's
    build_nx_graph(example, property_name=...) is what puts a real QM9
    property there instead of the usual dummy 0 (see that function's
    docstring). property_mean/property_std should be computed once
    from the training set (e.g. mu: mean=2.6520, std=1.4035 for this
    project's qm9_train.p) and passed in fixed -- normalizing the
    property onto a scale comparable to log(N+1) (roughly 0-2.3 for
    QM9's N range) matters here the same way it has everywhere else in
    this project (ActNorm's zero-init sensitivity, the conditional
    prior's sigma floor): an unnormalized raw-scale feature sitting
    next to log(N+1) would dominate the embedding purely from scale,
    not actual importance.
    """

    def __init__(self,
                embed_dim,
                property_mean,
                property_std,
                hidden_dim=64,
                name="NPropertyEmbedding"):
        super(NPropertyEmbedding, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.property_mean = property_mean
        self.property_std = property_std

    def _build(self, graph):
        n_node = tf.cast(tf.reshape(graph.n_node, [-1, 1]), tf.float32)
        n_feat = tf.log(n_node + 1.0)

        raw_property = tf.cast(tf.reshape(graph.globals, [-1, 1]), tf.float32)
        property_feat = (raw_property - self.property_mean) / self.property_std

        feat = tf.concat([n_feat, property_feat], axis=1)
        mlp = snt.nets.MLP(
            [self.hidden_dim, self.embed_dim],
            activation=tf.nn.relu,
            initializers={
                'w': tf.initializers.glorot_normal(),
                'b': tf.initializers.truncated_normal(stddev=0.1),
            },
            activate_final=False)
        return mlp(feat)
