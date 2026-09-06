"""Phase 2 correctness check: does the trained molecular auto-encoder's
*decoded, discrete* output actually correspond to a valid molecule, not
just individually-plausible atom/bond logits?

train_molecular_autoencoder.py's atom_accuracy/bond_accuracy are
per-node/per-pair classification accuracy -- they don't tell us whether
argmax-ing every prediction and reconstructing a molecule from them
produces something RDKit accepts as chemically valid, let alone the
right molecule. This is the molecular-autoencoder analogue of
check_roundtrip.py, which caught the BatchNorm invertibility bug earlier
in this project even though the flow's own loss/log-probs looked fine.

Compares two decoding strategies side by side:
  - naive: argmax every pair's bond-type logits independently. Nothing
    stops this from giving an atom more total bond order than its
    valence allows, since each pair is scored in isolation.
  - valence-aware: qm9_chem.decode_bonds_valence_aware's greedy,
    budget-respecting decode, built specifically to fix that failure
    mode without any retraining.

Rebuilds the model in the exact same construction order as
train_molecular_autoencoder.py (via the same molecular_reconstruction_loss
call) so variable names line up with the checkpoint. Read-only: restores
the checkpoint but never modifies it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import warnings

from absl import app
from absl import flags
import graph_nets as gn
import numpy as np
from rdkit import Chem
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gnn import TimestepGNN, dm_self_attn_gnn, make_mlp_model

from functools import partial

from bond_aware_attention import bond_aware_self_attn_gnn
from molecular_gnn import embed_atom_features, molecular_reconstruction_loss
from qm9_chem import decode_bonds_valence_aware, graph_to_mol
from qm9_graph_data import NUM_BOND_TYPES, build_nx_graph
from tf_helpers import reset_sess

warnings.filterwarnings("ignore")

flags.DEFINE_string('checkpoint', '', 'Trained autoencoder checkpoint '
                    'to restore, e.g. '
                    'molecular_generation/test_runs/smoke_test/'
                    'checkpoints-2000.')
flags.DEFINE_string('data_dir', 'molecular_generation/data', '')
flags.DEFINE_integer('num_molecules_to_check', 20, '')

# Must match the values used at training time exactly (Saver restores by
# variable name, but the graph must have the right shapes to build them).
flags.DEFINE_integer('node_embedding_dim', 32, '')
flags.DEFINE_integer('latent_dim', 128, '')
flags.DEFINE_integer('num_mlp_layers', 2, '')
flags.DEFINE_integer('attn_kq_dim', 32, '')
flags.DEFINE_integer('attn_v_dim', 32, '')
flags.DEFINE_integer('attn_num_heads', 2, '')
flags.DEFINE_integer('attn_concat_heads_output_dim', 32, '')
flags.DEFINE_integer('num_processing_steps', 4, '')
flags.DEFINE_bool('weight_sharing', True, '')
flags.DEFINE_bool('use_batch_norm', False, '')
flags.DEFINE_bool('residual', False, '')
flags.DEFINE_integer(
    'num_bond_refine_steps', 1,
    'Must match training -- more refine steps means more decoder '
    'variables in the checkpoint (each round has its own weights).')
flags.DEFINE_bool(
    'use_bond_aware_attention', False,
    'Must match training -- see bond_aware_attention.py / '
    'train_molecular_autoencoder.py\'s flag docstring.')

FLAGS = flags.FLAGS


def main(argv):
    del argv

    with open(os.path.join(FLAGS.data_dir, 'qm9_val.p'), 'rb') as f:
        val_examples = pickle.load(f)
    val_examples = val_examples[:FLAGS.num_molecules_to_check]
    print("Checking {} held-out val molecules against {}".format(
        len(val_examples), FLAGS.checkpoint))

    graphs = [build_nx_graph(e) for e in val_examples]

    raw_graph_phs = gn.utils_tf.placeholders_from_networkxs(
        graphs, force_dynamic_num_graphs=True, name="raw_graph_phs")
    raw_graph_phs.n_node.set_shape([len(graphs)])

    embedded_graph_phs = embed_atom_features(
        raw_graph_phs, FLAGS.latent_dim, FLAGS.node_embedding_dim,
        FLAGS.num_mlp_layers)

    make_mlp_fn = partial(make_mlp_model, FLAGS.latent_dim,
                          FLAGS.node_embedding_dim, FLAGS.num_mlp_layers)
    if FLAGS.use_bond_aware_attention:
        attn_gnn_fn = partial(
            bond_aware_self_attn_gnn,
            kq_dim=FLAGS.attn_kq_dim,
            v_dim=FLAGS.attn_v_dim,
            make_mlp_fn=make_mlp_fn,
            num_heads=FLAGS.attn_num_heads,
            concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
            num_bond_types=NUM_BOND_TYPES,
            concat=True,
            residual=False,
            layer_norm=False,
            kq_dim_division=True)
    else:
        attn_gnn_fn = partial(
            dm_self_attn_gnn,
            kq_dim=FLAGS.attn_kq_dim,
            v_dim=FLAGS.attn_v_dim,
            make_mlp_fn=make_mlp_fn,
            num_heads=FLAGS.attn_num_heads,
            concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
            concat=True,
            residual=False,
            layer_norm=False,
            kq_dim_division=True)

    is_training = tf.placeholder(tf.bool, name="is_training")
    gnn = TimestepGNN(
        attn_gnn_fn,
        FLAGS.num_processing_steps,
        weight_sharing=FLAGS.weight_sharing,
        use_batch_norm=FLAGS.use_batch_norm,
        residual=FLAGS.residual,
        test_local_stats=True)
    gnn_output = gnn(embedded_graph_phs, is_training=is_training)

    losses = molecular_reconstruction_loss(
        raw_graph_phs, gnn_output, FLAGS.node_embedding_dim,
        FLAGS.latent_dim, FLAGS.num_mlp_layers,
        num_bond_refine_steps=FLAGS.num_bond_refine_steps)

    sess = reset_sess()
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.checkpoint)

    values = sess.run(
        {
            'atom_pred': losses['atom_pred'],
            'bond_pred': losses['bond_pred'],
            'bond_probs': losses['bond_probs'],
            'atom_accuracy': losses['atom_accuracy'],
            'bond_accuracy': losses['bond_accuracy'],
        },
        feed_dict={raw_graph_phs: gn.utils_np.networkxs_to_graphs_tuple(graphs),
                  is_training: False})

    print("Per-node/per-pair accuracy on this batch: "
         "atom_accuracy={:.4f} bond_accuracy={:.4f}".format(
             values['atom_accuracy'], values['bond_accuracy']))

    def try_reconstruct(atom_pred, bond_matrix):
        """Returns (recon_smiles or None, error or None)."""
        try:
            recon_mol = graph_to_mol(atom_pred, bond_matrix)
        except Exception as e:
            return None, e
        return Chem.MolToSmiles(recon_mol), None

    n_nodes = [e['n_node'] for e in val_examples]
    n_node_cum = np.cumsum(n_nodes)
    start = 0
    naive_valid = naive_match = 0
    va_valid = va_match = 0
    for i, example in enumerate(val_examples):
        end = n_node_cum[i]
        atom_pred = values['atom_pred'][start:end].tolist()
        naive_bonds = values['bond_pred'][start:end, start:end]
        bond_probs = values['bond_probs'][start:end, start:end]
        start = end

        orig_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(example['smiles']))

        naive_smiles, naive_err = try_reconstruct(atom_pred, naive_bonds)
        va_bonds = decode_bonds_valence_aware(atom_pred, bond_probs)
        va_smiles, va_err = try_reconstruct(atom_pred, va_bonds)

        naive_valid += int(naive_smiles is not None)
        naive_match += int(naive_smiles == orig_canonical)
        va_valid += int(va_smiles is not None)
        va_match += int(va_smiles == orig_canonical)

        def describe(smiles, err):
            if smiles is None:
                return "INVALID ({})".format(err)
            return "MATCH" if smiles == orig_canonical else "valid-but-different"

        print("  [{}] orig={}".format(i, orig_canonical))
        print("      naive:          {:<24} recon={}".format(
            describe(naive_smiles, naive_err), naive_smiles))
        print("      valence-aware:  {:<24} recon={}".format(
            describe(va_smiles, va_err), va_smiles))

    n = len(val_examples)
    print("\n" + "=" * 60)
    print("{:<20} {:>18} {:>18}".format("", "naive argmax", "valence-aware"))
    print("{:<20} {:>15}/{:<3}({:>5.1f}%) {:>15}/{:<3}({:>5.1f}%)".format(
        "Valid reconstructions:", naive_valid, n, 100.0 * naive_valid / n,
        va_valid, n, 100.0 * va_valid / n))
    print("{:<20} {:>15}/{:<3}({:>5.1f}%) {:>15}/{:<3}({:>5.1f}%)".format(
        "Exact matches:", naive_match, n, 100.0 * naive_match / n,
        va_match, n, 100.0 * va_match / n))
    print("=" * 60)


if __name__ == '__main__':
    app.run(main)
