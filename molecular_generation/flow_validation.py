"""Held-out validation loss and early stopping for the molecular flow
trainers. Deliberately free of TensorFlow/absl imports so it can be unit
tested without the cluster environment.

The validation set is the frozen Phase 2 encoder's embeddings of Phase 1's
val split (generate_molecular_embeddings.py --split val). The test split is
never touched.
"""

import os
import pickle

import numpy as np


class ValidationSet(object):
    """Loads every embeddings_*.p file in val_data_dir and serves
    exact-size batches (the flow's n_node placeholder has a fixed batch
    size, so the final partial batch is dropped).
    """

    def __init__(self, val_data_dir, batch_size, max_batches=0, max_files=0):
        files = sorted(f for f in os.listdir(val_data_dir)
                       if f.startswith('embeddings_'))
        if max_files > 0:
            files = files[:max_files]
        if not files:
            raise ValueError(
                "No embeddings_*.p files in {}. Run "
                "generate_molecular_embeddings.py --split val first.".format(
                    val_data_dir))
        node_chunks, n_node_chunks = [], []
        for name in files:
            with open(os.path.join(val_data_dir, name), 'rb') as f:
                d = pickle.load(f)
            node_chunks.append(d[0])
            n_node_chunks.append(d[1])
        self.node_embeddings = np.concatenate(node_chunks, axis=0)
        self.n_node = np.concatenate(n_node_chunks, axis=0)
        self.n_node_cs = np.concatenate([[0], np.cumsum(self.n_node)])
        self.batch_size = batch_size
        num_batches = len(self.n_node) // batch_size
        if max_batches > 0:
            num_batches = min(num_batches, max_batches)
        self.num_batches = num_batches

    @property
    def num_molecules(self):
        return self.num_batches * self.batch_size

    def batches(self):
        bs = self.batch_size
        for b in range(self.num_batches):
            lo, hi = b * bs, (b + 1) * bs
            yield (self.node_embeddings[self.n_node_cs[lo]:self.n_node_cs[hi]],
                   self.n_node[lo:hi])

    def evaluate(self, sess, node_embeddings_placeholder, n_node_placeholder,
                 batch_sum_tensors, extra_feed=None):
        """batch_sum_tensors: {name: scalar tensor that is a SUM over the
        batch} (total_loss, log_prob_zs, log_det_jacobian). Returns
        {name: that sum divided by the total node count over the whole
        validation set}, i.e. directly comparable to the training
        per_node_loss. extra_feed: additional placeholders fed on every
        batch (e.g. switching training-only noise off).
        """
        sums = {k: 0.0 for k in batch_sum_tensors}
        num_nodes = 0
        for node_embeddings, n_node in self.batches():
            feed = dict(extra_feed or {})
            feed[node_embeddings_placeholder] = node_embeddings
            feed[n_node_placeholder] = n_node
            values = sess.run(batch_sum_tensors, feed_dict=feed)
            for k, v in values.items():
                sums[k] += float(v)
            num_nodes += int(np.sum(n_node))
        return {k: v / num_nodes for k, v in sums.items()}


class EarlyStopper(object):
    """Stops when the validation loss has not improved by more than
    min_delta for `patience` consecutive evaluations (Keras semantics:
    the reference value only moves on a qualifying improvement, so a slow
    drift of many tiny gains still counts as improving). The checkpoint
    to keep is tracked separately and is the lowest validation loss seen,
    however small the improvement. patience=0 disables stopping.
    """

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.best_step = None
        self._reference = float('inf')
        self.evals_since_improvement = 0

    def update(self, value, step):
        """Returns (is_new_best, should_stop)."""
        is_new_best = value < self.best
        if is_new_best:
            self.best = value
            self.best_step = step
        if value < self._reference - self.min_delta:
            self._reference = value
            self.evals_since_improvement = 0
        else:
            self.evals_since_improvement += 1
        should_stop = (self.patience > 0 and
                       self.evals_since_improvement >= self.patience)
        return is_new_best, should_stop
