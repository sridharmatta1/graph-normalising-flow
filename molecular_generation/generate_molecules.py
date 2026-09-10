"""Phase 6: generates novel molecules end to end.

Samples fresh embeddings from Phase 5's trained flow (conditioned on N
values drawn from QM9's own empirical N distribution, since
N-conditioning -- Phase 4 -- is still deliberately deferred), decodes
them through Phase 2's frozen decoder (atom-type + bond-type heads,
then qm9_chem.py's valence-aware constrained decoding), and reports
the standard molecular generation metrics: validity, uniqueness, and
novelty (vs. the training set) -- the molecular analogue of community/
ego's degree/clustering/orbit MMD.

Restores the two checkpoints (Phase 2's frozen encoder/decoder, Phase
5's flow) into two ENTIRELY SEPARATE tf.Graph/tf.Session pairs, with
numpy arrays as the only thing crossing the boundary between them.
Both checkpoints were saved with sonnet's default auto-incrementing
variable names (no explicit scope), so building them one after another
in the SAME graph would silently rename the second model's variables
(e.g. "mlp_0" colliding with a name the first model already claimed),
breaking restoration -- two independent graphs sidestep that entirely,
since each is built in exactly the order its own training script used.

Each graph is rebuilt by calling the same construction functions its
training script used, in the same order:
  - Phase 2 graph: embed_atom_features -> TimestepGNN (bond-aware
    attention) -> atom_type_logits -> refine_bond_logits, exactly as
    train_molecular_autoencoder.py did. The encoder's own forward pass
    is rebuilt on a small dummy batch purely so its modules' variables
    get created in the right order/names -- its output is never used;
    the decoder heads are then called again on the flow's generated
    embeddings instead.
  - Phase 5 graph: MolecularGNFBlock's f() (dummy call, any batch)
    THEN g() (the real sampling call) -- training only ever called
    f(), and f() is what creates every one of the 24 independent
    coupling sub-networks' variables, in a specific traversal order;
    calling g() alone, fresh, would traverse them in a different order
    and silently produce different (but shape-compatible) variable
    names, which would restore without error but assign the wrong
    weights to the wrong sub-network.

Untested against a real TF graph before reaching the cluster, same as
every other script in this project -- expect a first-run fix or two.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import json
import os
import pickle
import random
import sys
import warnings

from absl import app
from absl import flags
import graph_nets as gn
import numpy as np
from rdkit import Chem
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gnn import TimestepGNN, dm_self_attn_gnn, make_mlp_model

from bond_aware_attention import bond_aware_self_attn_gnn
from molecular_flow import MolecularGNFBlock
from molecular_gnn import (atom_type_logits, bond_type_logits,
                          embed_atom_features, refine_bond_logits)
from qm9_chem import decode_bonds_valence_aware, graph_to_mol
from qm9_graph_data import NUM_BOND_TYPES, build_nx_graph
from tf_helpers import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

flags.DEFINE_string(
    'phase2_checkpoint',
    'molecular_generation/checkpoints/phase2_final/checkpoints-50001', '')
flags.DEFINE_string(
    'flow_checkpoint',
    'molecular_generation/test_runs/flow_full/checkpoints-100001', '')
flags.DEFINE_string('data_dir', 'molecular_generation/data', '')
flags.DEFINE_integer('num_molecules_to_generate', 200, '')
flags.DEFINE_integer('sample_batch_size', 32,
                     'Molecules decoded together per batch -- keeps '
                     'the O(N^2) pairwise bond-decoder call from '
                     'growing too large in one shot.')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_string(
    'save_results_json', '',
    'If non-empty, write every generated molecule (SMILES, connected, '
    'novel) plus the summary metrics to this JSON path -- for '
    'visualize_molecules.py to render afterwards, without needing to '
    're-run generation (which requires both checkpoints) just to look '
    'at the structures.')

# Shared between both checkpoints -- the flow's input/output IS Phase
# 2's encoder-output embedding space, so these must match exactly.
flags.DEFINE_integer('node_embedding_dim', 64, '')

# Phase 2 architecture (must match phase2_checkpoint's training flags --
# see run_molecular_autoencoder_v5.sh).
flags.DEFINE_integer('p2_latent_dim', 256, '')
flags.DEFINE_integer('p2_num_mlp_layers', 2, '')
flags.DEFINE_integer('p2_num_processing_steps', 6, '')
flags.DEFINE_integer('p2_attn_kq_dim', 32, '')
flags.DEFINE_integer('p2_attn_v_dim', 32, '')
flags.DEFINE_integer('p2_attn_num_heads', 2, '')
flags.DEFINE_integer('p2_attn_concat_heads_output_dim', 32, '')
flags.DEFINE_bool('p2_weight_sharing', True, '')
flags.DEFINE_integer('p2_num_bond_refine_steps', 1, '')

# Phase 5 architecture (must match flow_checkpoint's training flags --
# see run_molecular_flow_full.sh).
flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_integer('flow_latent_dim', 512, '')
flags.DEFINE_integer('flow_num_layers', 3, '')
flags.DEFINE_integer('flow_attn_kq_dim', 64, '')
flags.DEFINE_integer('flow_attn_v_dim', 64, '')
flags.DEFINE_integer('flow_attn_num_heads', 2, '')
flags.DEFINE_integer('flow_attn_concat_heads_output_dim', 64, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')

FLAGS = flags.FLAGS


def transform_example(n_node):
    globals_ = tf.zeros_like(n_node)
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, globals_, receivers, senders, n_edge


def build_flow_graph():
    """Restores Phase 5's flow into its own Graph/Session. Returns
    (sess, sample_n_node_placeholder, generated_embeddings_tensor).
    """
    graph = tf.Graph()
    with graph.as_default():
        node_embeddings_placeholder = tf.placeholder(
            tf.float32, shape=[None, FLAGS.node_embedding_dim],
            name='node_embeddings_placeholder')
        n_node_placeholder = tf.placeholder(tf.int32, shape=[None],
                                            name='n_node_placeholder')
        edges, globals_, receivers, senders, n_edge = transform_example(
            n_node_placeholder)
        graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                             edges=edges,
                                             globals=globals_,
                                             receivers=receivers,
                                             senders=senders,
                                             n_node=n_node_placeholder,
                                             n_edge=n_edge)

        half_dim = FLAGS.node_embedding_dim // 2
        make_mlp_fn = partial(make_mlp_model, FLAGS.flow_latent_dim,
                              half_dim, FLAGS.flow_num_layers)
        attn_gnn_fn = partial(
            dm_self_attn_gnn,
            kq_dim=FLAGS.flow_attn_kq_dim,
            v_dim=FLAGS.flow_attn_v_dim,
            make_mlp_fn=make_mlp_fn,
            num_heads=FLAGS.flow_attn_num_heads,
            concat_heads_output_dim=FLAGS.flow_attn_concat_heads_output_dim,
            concat=True,
            residual=False,
            layer_norm=False,
            kq_dim_division=True)
        grevnet = MolecularGNFBlock(attn_gnn_fn, FLAGS.flow_num_coupling_layers,
                                    half_dim, FLAGS.flow_weight_sharing,
                                    FLAGS.flow_max_log_scale)

        # Dummy f() call: never run, only needed so every one of the 24
        # independent coupling sub-networks' variables get created now,
        # in the same traversal order training used -- g() alone,
        # called first and fresh, would create them in a different
        # order and silently mismatch the checkpoint.
        grevnet(graphs_tuple, inverse=True)

        sample_n_node_placeholder = tf.placeholder(
            tf.int32, shape=[None], name='sample_n_node_placeholder')
        mvn = tfd.MultivariateNormalDiag(
            tf.zeros(FLAGS.node_embedding_dim),
            tf.ones(FLAGS.node_embedding_dim))
        sample_nodes = mvn.sample(
            sample_shape=(tf.reduce_sum(sample_n_node_placeholder),))
        s_edges, s_globals, s_receivers, s_senders, s_n_edge = transform_example(
            sample_n_node_placeholder)
        sample_graphs_tuple = gn.graphs.GraphsTuple(
            nodes=sample_nodes, edges=s_edges, globals=s_globals,
            receivers=s_receivers, senders=s_senders,
            n_node=sample_n_node_placeholder, n_edge=s_n_edge)

        generated = grevnet(sample_graphs_tuple, inverse=False)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring Phase 5 flow from {}".format(FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)

    return sess, sample_n_node_placeholder, generated.nodes


def build_decoder_graph():
    """Restores Phase 2's frozen encoder/decoder into its own Graph/
    Session. Returns (sess, generated_embeddings_placeholder,
    atom_pred_tensor, bond_probs_tensor).
    """
    graph = tf.Graph()
    with graph.as_default():
        with open(os.path.join(FLAGS.data_dir, 'qm9_train.p'), 'rb') as f:
            train_examples = pickle.load(f)
        dummy_graphs = [build_nx_graph(e) for e in train_examples[:4]]

        raw_graph_phs = gn.utils_tf.placeholders_from_networkxs(
            dummy_graphs, force_dynamic_num_graphs=True,
            name="raw_graph_phs")

        embedded_graph_phs = embed_atom_features(
            raw_graph_phs, FLAGS.p2_latent_dim, FLAGS.node_embedding_dim,
            FLAGS.p2_num_mlp_layers)

        make_mlp_fn = partial(make_mlp_model, FLAGS.p2_latent_dim,
                              FLAGS.node_embedding_dim,
                              FLAGS.p2_num_mlp_layers)
        attn_gnn_fn = partial(
            bond_aware_self_attn_gnn,
            kq_dim=FLAGS.p2_attn_kq_dim,
            v_dim=FLAGS.p2_attn_v_dim,
            make_mlp_fn=make_mlp_fn,
            num_heads=FLAGS.p2_attn_num_heads,
            concat_heads_output_dim=FLAGS.p2_attn_concat_heads_output_dim,
            num_bond_types=NUM_BOND_TYPES,
            concat=True,
            residual=False,
            layer_norm=False,
            kq_dim_division=True)

        is_training = tf.placeholder(tf.bool, name="is_training")
        gnn = TimestepGNN(attn_gnn_fn, FLAGS.p2_num_processing_steps,
                          weight_sharing=FLAGS.p2_weight_sharing,
                          use_batch_norm=False, residual=False,
                          test_local_stats=True)
        # Dummy encoder pass: never run, only needed so the encoder's
        # own attention modules' variables get created now, in the
        # same order training used, before the decoder heads below.
        gnn(embedded_graph_phs, is_training=is_training)

        generated_embeddings_placeholder = tf.placeholder(
            tf.float32, shape=[None, FLAGS.node_embedding_dim],
            name='generated_embeddings_placeholder')
        fake_gnn_output = raw_graph_phs.replace(
            nodes=generated_embeddings_placeholder)

        atom_logits = atom_type_logits(fake_gnn_output, FLAGS.p2_latent_dim,
                                       FLAGS.p2_num_mlp_layers)
        atom_pred = tf.argmax(atom_logits, axis=1, output_type=tf.int32)

        bond_logits = refine_bond_logits(
            fake_gnn_output, raw_graph_phs, atom_pred,
            FLAGS.node_embedding_dim, FLAGS.p2_latent_dim,
            FLAGS.p2_num_mlp_layers,
            num_refine_steps=FLAGS.p2_num_bond_refine_steps)
        bond_probs = tf.nn.softmax(bond_logits, axis=-1)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring Phase 2 decoder from {}".format(
            FLAGS.phase2_checkpoint))
        saver.restore(sess, FLAGS.phase2_checkpoint)

    return sess, generated_embeddings_placeholder, atom_pred, bond_probs


def main(argv):
    del argv
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    with open(os.path.join(FLAGS.data_dir, 'qm9_train.p'), 'rb') as f:
        train_examples = pickle.load(f)
    train_n_node = [e['n_node'] for e in train_examples]
    train_smiles = set(
        Chem.MolToSmiles(Chem.MolFromSmiles(e['smiles']))
        for e in train_examples)
    print("Loaded {} training molecules (for N-sampling and novelty "
         "check)".format(len(train_examples)))

    flow_sess, sample_n_node_ph, generated_embeddings_t = build_flow_graph()
    decoder_sess, embeddings_ph, atom_pred_t, bond_probs_t = build_decoder_graph()

    results = []  # list of (canonical_smiles or None, is_connected or None)
    num_generated = 0
    while num_generated < FLAGS.num_molecules_to_generate:
        batch_size = min(FLAGS.sample_batch_size,
                         FLAGS.num_molecules_to_generate - num_generated)
        batch_n_node = [random.choice(train_n_node) for _ in range(batch_size)]

        embeddings = flow_sess.run(
            generated_embeddings_t,
            feed_dict={sample_n_node_ph: batch_n_node})

        atom_pred, bond_probs = decoder_sess.run(
            [atom_pred_t, bond_probs_t],
            feed_dict={embeddings_ph: embeddings})

        n_node_cum = np.cumsum(batch_n_node)
        start = 0
        for i, n in enumerate(batch_n_node):
            end = n_node_cum[i]
            mol_atom_pred = atom_pred[start:end].tolist()
            mol_bond_probs = bond_probs[start:end, start:end]
            start = end

            bond_matrix = decode_bonds_valence_aware(mol_atom_pred,
                                                     mol_bond_probs)
            try:
                mol = graph_to_mol(mol_atom_pred, bond_matrix)
                smiles = Chem.MolToSmiles(mol)
                is_connected = len(Chem.GetMolFrags(mol)) == 1
            except Exception:
                smiles = None
                is_connected = None
            results.append((smiles, is_connected))

        num_generated += batch_size
        print("Generated {}/{} molecules".format(
            num_generated, FLAGS.num_molecules_to_generate))

    valid = [s for s, c in results if s is not None]
    connected = [s for s, c in results if s is not None and c]
    # RDKit sanitization (validity) only checks each fragment's own
    # valence -- it says nothing about whether the result is a SINGLE
    # connected molecule. QM9's real molecules never have floating
    # isolated atoms, so "valid" alone overstates how many outputs are
    # genuinely well-formed single molecules; "connected" is the
    # stricter, more honest number.
    valid_unique = set(valid)
    valid_novel = [s for s in valid_unique if s not in train_smiles]
    connected_unique = set(connected)
    connected_novel = [s for s in connected_unique if s not in train_smiles]

    n = len(results)
    validity = len(valid) / n
    connectivity = (len(connected) / len(valid)) if valid else 0.0
    strict_validity = len(connected) / n
    valid_uniqueness = (len(valid_unique) / len(valid)) if valid else 0.0
    valid_novelty = ((len(valid_novel) / len(valid_unique))
                     if valid_unique else 0.0)
    connected_uniqueness = ((len(connected_unique) / len(connected))
                           if connected else 0.0)
    connected_novelty = ((len(connected_novel) / len(connected_unique))
                        if connected_unique else 0.0)

    print("\n" + "=" * 60)
    print("Generated {} molecules".format(n))
    print("Validity (RDKit-sanitizable, fragments allowed): "
         "{}/{} ({:.1f}%)".format(len(valid), n, 100 * validity))
    print("  Uniqueness (of valid): {}/{} ({:.1f}%)".format(
        len(valid_unique), len(valid), 100 * valid_uniqueness))
    print("  Novelty (of unique valid, vs. training set): "
         "{}/{} ({:.1f}%)".format(len(valid_novel), len(valid_unique),
                                  100 * valid_novelty))
    print("-" * 60)
    print("Connectivity (of valid, single connected component): "
         "{}/{} ({:.1f}%)".format(len(connected), len(valid),
                                  100 * connectivity))
    print("Strict validity (valid AND connected -- a real single "
         "molecule): {}/{} ({:.1f}%)".format(len(connected), n,
                                             100 * strict_validity))
    print("  Uniqueness (of strictly valid): {}/{} ({:.1f}%)".format(
        len(connected_unique), len(connected), 100 * connected_uniqueness))
    print("  Novelty (of unique strictly-valid, vs. training set): "
         "{}/{} ({:.1f}%)".format(len(connected_novel), len(connected_unique),
                                  100 * connected_novelty))
    print("=" * 60)
    print("\nSample of strictly-valid generated molecules:")
    for s in list(connected_unique)[:20]:
        print(" ", s)

    if FLAGS.save_results_json:
        molecules = []
        for smiles, is_connected in results:
            molecules.append({
                'smiles': smiles,
                'connected': bool(is_connected) if is_connected is not None else False,
                'novel': (smiles in connected_novel) if
                         (smiles is not None and is_connected) else False,
            })
        payload = {
            'summary': {
                'num_generated': n,
                'validity': validity,
                'connectivity': connectivity,
                'strict_validity': strict_validity,
                'valid_uniqueness': valid_uniqueness,
                'valid_novelty': valid_novelty,
                'connected_uniqueness': connected_uniqueness,
                'connected_novelty': connected_novelty,
            },
            'molecules': molecules,
        }
        os.makedirs(os.path.dirname(FLAGS.save_results_json) or '.',
                   exist_ok=True)
        with open(FLAGS.save_results_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print("\nSaved {} generated molecules + summary metrics to {}".format(
            n, FLAGS.save_results_json))


if __name__ == '__main__':
    app.run(main)
