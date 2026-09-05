"""Phase 2, part 1: turns Phase 1's (atom_types, bond_matrix) pickles into
the networkx-graph-batch interface run_gnn.py's training loop expects
(GraphDataset's public methods), so the existing TimestepGNN encoder and
training-loop machinery can be reused almost unchanged.

Key difference from graph_data.py's GraphDataset: node features here are
real one-hot atom types (not random Gaussian noise), and edges carry the
real bond-type index as a feature (not always 0) -- self-loops are still
added (every node gets an (i, i) edge with a dummy 0 feature) because the
encoder's message passing needs them to let a node attend to itself, same
as graph_data.py's convert_nx_repr does for structural graphs. Bond type
on the diagonal is never used (the molecular_gnn.py losses explicitly mask
the diagonal out), so its dummy value doesn't matter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random

import graph_nets as gn
import networkx as nx
import numpy as np

from qm9_constants import ATOM_VOCAB

NUM_ATOM_TYPES = len(ATOM_VOCAB)
# 0=none (implicit -- never an explicit edge), 1=single, 2=double, 3=triple.
NUM_BOND_TYPES = 4


def build_nx_graph(example):
    """One Phase-1 example dict -> a directed networkx graph matching
    graph_data.py's convert_nx_repr conventions (self-loops included,
    graph-level 'features' key present) so networkxs_to_graphs_tuple /
    placeholders_from_networkxs work unmodified.
    """
    atom_types = example['atom_types']
    bond_matrix = example['bond_matrix']
    n = len(atom_types)

    g = nx.DiGraph(features=0)
    for i in range(n):
        one_hot = np.zeros(NUM_ATOM_TYPES, dtype=np.float32)
        one_hot[atom_types[i]] = 1.0
        g.add_node(i, features=one_hot)
        g.add_edge(i, i, features=np.array([0.0], dtype=np.float32))
    for i in range(n):
        for j in range(n):
            if i != j and bond_matrix[i, j] != 0:
                g.add_edge(
                    i, j,
                    features=np.array([float(bond_matrix[i, j])],
                                      dtype=np.float32))
    return g


class QM9GraphDataset():
    """Mirrors graph_data.py's GraphDataset public interface (train_graphs,
    get_next_train_batch, get_random_test_batch, train_n_nodes) so
    train_molecular_autoencoder.py's training loop looks like run_gnn.py's.

    Uses Phase 1's val split as the held-out "test" set here, and never
    touches the test split -- keeping a final, never-peeked-at set for
    later evaluation once the full pipeline (Phases 2-6) is done.
    """

    def __init__(self, data_dir, max_molecules=0):
        with open(os.path.join(data_dir, 'qm9_train.p'), 'rb') as f:
            train_examples = pickle.load(f)
        with open(os.path.join(data_dir, 'qm9_val.p'), 'rb') as f:
            val_examples = pickle.load(f)

        if max_molecules > 0:
            train_examples = train_examples[:max_molecules]
            val_examples = val_examples[:max(1, max_molecules // 10)]

        self.train_graphs = [build_nx_graph(e) for e in train_examples]
        self.test_graphs = [build_nx_graph(e) for e in val_examples]
        self.train_index = 0
        self.test_index = 0

    def train_n_nodes(self):
        return [g.number_of_nodes() for g in self.train_graphs]

    def test_n_nodes(self):
        return [g.number_of_nodes() for g in self.test_graphs]

    def get_next_train_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            if self.train_index == 0:
                random.shuffle(self.train_graphs)
            batch.append(self.train_graphs[self.train_index])
            self.train_index = (self.train_index + 1) % len(self.train_graphs)
        return gn.utils_np.networkxs_to_graphs_tuple(batch)

    def get_random_test_batch(self, batch_size):
        return gn.utils_np.networkxs_to_graphs_tuple(
            random.choices(self.test_graphs, k=batch_size))
