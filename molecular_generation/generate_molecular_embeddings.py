"""Phase 3, part 1: extracts node embeddings from the frozen Phase 2
encoder, over the whole training set -- the molecular analogue of
generate_grevnet_training_data.py, which does the same thing for
community/ego's encoder. These embeddings become the GNF flow's own
training data (Phase 5): the flow never sees real molecules directly,
only the (fixed, no-longer-trained) encoder's embeddings of them.

Restores the checkpoint via import_meta_graph + feeding placeholders by
name (rather than rebuilding the graph in Python, like
check_autoencoder_roundtrip.py does) -- same pattern
generate_grevnet_training_data.py already uses, and it sidesteps having
to know this checkpoint's exact hyperparameters (node_embedding_dim,
num_processing_steps, use_bond_aware_attention, ...) here at all: the
meta graph already encodes the full built architecture.

Only ever run this against the frozen checkpoint in
molecular_generation/checkpoints/phase2_final/ -- not a test_runs/
checkpoint that might still be overwritten by a future training run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import warnings

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from qm9_graph_data import QM9GraphDataset
from tf_helpers import reset_sess

warnings.filterwarnings("ignore")

flags.DEFINE_string(
    'checkpoint', 'molecular_generation/checkpoints/phase2_final/checkpoints-50001',
    'Frozen Phase 2 checkpoint to restore.')
flags.DEFINE_string('data_dir', 'molecular_generation/data', '')
flags.DEFINE_string('output_file', 'molecular_generation/data/embeddings',
                    '')
flags.DEFINE_integer(
    'node_embedding_dim', 64,
    'Must match the checkpoint\'s --node_embedding_dim (only used here '
    'to size the output array / chunk-size check, not to rebuild the '
    'graph -- import_meta_graph restores the architecture as-is).')
flags.DEFINE_integer('train_batch_size', 32,
                     'Must match the checkpoint\'s training batch size '
                     '-- the restored graph\'s placeholders have that '
                     'exact batch size baked into their static shape.')
flags.DEFINE_integer(
    'num_examples', 0,
    'How many molecules to extract embeddings for. 0 = the full '
    'training set (one pass, rounded up to a multiple of '
    'train_batch_size).')
flags.DEFINE_integer('run_number', 0, '')
FLAGS = flags.FLAGS


def main(argv):
    del argv
    dataset = QM9GraphDataset(FLAGS.data_dir)
    num_examples_target = FLAGS.num_examples or len(dataset.train_graphs)
    print("Extracting embeddings for {} molecules ({} available) from {}"
         .format(num_examples_target, len(dataset.train_graphs),
                 FLAGS.checkpoint))

    sess = reset_sess()
    saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.checkpoint))
    saver.restore(sess, FLAGS.checkpoint)

    values_map = {
        'gnn_output_nodes': tf.get_collection('gnn_output_nodes')[0],
        'atom_accuracy': tf.get_collection('atom_accuracy')[0],
        'bond_accuracy': tf.get_collection('bond_accuracy')[0],
    }

    filename_template = "{}_{}_{{}}.p".format(FLAGS.output_file,
                                              FLAGS.run_number)
    file_number = 0
    filename = filename_template.format(file_number)

    total_n_node = 0
    node_embeddings = np.empty([0, FLAGS.node_embedding_dim])
    n_node = np.empty([0], dtype=np.int32)

    num_examples = 0
    batch_num = 0
    while num_examples < num_examples_target:
        if total_n_node * FLAGS.node_embedding_dim * 4 > 100e6:
            with open(filename, 'wb') as f:
                pickle.dump((node_embeddings, n_node), f)
            print("Wrote {} molecules to {}".format(len(n_node), filename))
            file_number += 1
            filename = filename_template.format(file_number)
            total_n_node = 0
            node_embeddings = np.empty([0, FLAGS.node_embedding_dim])
            n_node = np.empty([0], dtype=np.int32)

        graphs_tuple = dataset.get_next_train_batch(FLAGS.train_batch_size)
        feed_dict = {
            "raw_graph_phs/nodes:0": graphs_tuple.nodes,
            "raw_graph_phs/edges:0": graphs_tuple.edges,
            "raw_graph_phs/receivers:0": graphs_tuple.receivers,
            "raw_graph_phs/senders:0": graphs_tuple.senders,
            "raw_graph_phs/globals:0": graphs_tuple.globals,
            "raw_graph_phs/n_node:0": graphs_tuple.n_node,
            "raw_graph_phs/n_edge:0": graphs_tuple.n_edge,
            "is_training:0": False,
        }
        values = sess.run(values_map, feed_dict=feed_dict)

        n_node = np.append(n_node, graphs_tuple.n_node, axis=0)
        node_embeddings = np.append(node_embeddings,
                                    values['gnn_output_nodes'], axis=0)
        total_n_node += np.sum(graphs_tuple.n_node)
        num_examples += FLAGS.train_batch_size

        if batch_num % 100 == 0:
            print("batch {}, {} molecules so far -- "
                 "atom_accuracy={:.4f} bond_accuracy={:.4f}".format(
                     batch_num, num_examples, values['atom_accuracy'],
                     values['bond_accuracy']))
        batch_num += 1

    if len(n_node) > 0:
        with open(filename, 'wb') as f:
            pickle.dump((node_embeddings, n_node), f)
        print("Wrote {} molecules to {}".format(len(n_node), filename))

    print("Done. Extracted embeddings for {} molecules total.".format(
        num_examples))


if __name__ == '__main__':
    app.run(main)
