"""A condition-free stand-in for NEmbedding, used by the ablation that asks
whether N-conditioning itself (as opposed to the FiLM coupling networks it
comes bundled with) is responsible for the lower uniqueness of the
N-conditioned molecular flow.

NullEmbedding returns the same constant [num_graphs, embed_dim] vector of
zeros for every molecule, whatever its N. Plugged into
NConditionedMolecularGNFBlock (via embedding_module) and into the prior's
embedding, it leaves the FiLM architecture, the width, the learned diagonal
prior and every hyperparameter exactly as in the conditioned flow, while the
network can no longer see N. The FiLM generators and the prior head then
learn a single constant gamma/beta and a single (mu, sigma) for all sizes.
It has no variables, so checkpoints of an ablated flow simply lack the
NEmbedding weights.
"""

import sonnet as snt
import tensorflow as tf


class NullEmbedding(snt.AbstractModule):

    def __init__(self, embed_dim, name="NullEmbedding"):
        super(NullEmbedding, self).__init__(name=name)
        self.embed_dim = embed_dim

    def _build(self, graph):
        num_graphs = tf.shape(graph.n_node)[0]
        out = tf.zeros([num_graphs, self.embed_dim], dtype=tf.float32)
        out.set_shape([None, self.embed_dim])
        return out
