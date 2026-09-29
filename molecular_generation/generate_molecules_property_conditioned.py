"""Phase 6, property-conditioned variant: generates molecules from the
jointly N+property-conditioned flow (train_molecular_flow_property_conditioned.py),
with --target_n and --target_property to request both a specific
heavy-atom count AND a specific value of the conditioning property
(default: mu, dipole moment) at once.

Mirrors generate_molecules_n_conditioned.py closely -- same two-graph/
two-session restoration (Phase 2's frozen decoder is completely
unaffected by what Phase 5's flow was conditioned on), same
--target_n semantics. The only new piece is --target_property and
building the sampling graph's conditioning input from BOTH N and the
property value, via NPropertyEmbedding instead of plain NEmbedding.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from conditional_prior import ConditionalPriorNetwork, sample_conditional_prior
from gnn import TimestepGNN, make_mlp_model

from bond_aware_attention import bond_aware_self_attn_gnn
from molecular_flow import NConditionedMolecularGNFBlock
from molecular_gnn import (atom_type_logits, embed_atom_features,
                          refine_bond_logits)
from property_conditioning import NPropertyEmbedding
from qm9_chem import decode_bonds_valence_aware, graph_to_mol
from qm9_graph_data import NUM_BOND_TYPES, build_nx_graph
from tf_helpers import reset_sess, senders_receivers

warnings.filterwarnings("ignore")

flags.DEFINE_string(
    'phase2_checkpoint',
    'molecular_generation/checkpoints/phase2_final/checkpoints-50001', '')
flags.DEFINE_string(
    'flow_checkpoint',
    'molecular_generation/test_runs/flow_property_conditioned/checkpoints-100001',
    '')
flags.DEFINE_string('data_dir', 'molecular_generation/data', '')
flags.DEFINE_integer('num_molecules_to_generate', 200, '')
flags.DEFINE_integer('sample_batch_size', 32, '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_integer(
    'target_n', 0,
    'If > 0, force every generated molecule to have exactly this many '
    'heavy atoms. 0 (default) samples N from the training distribution, '
    'same as generate_molecules_n_conditioned.py.')
flags.DEFINE_float(
    'target_property', float('nan'),
    'The conditioning property value (default: mu, dipole moment) to '
    'request for every generated molecule. Required -- there is no '
    'sensible "sample from training distribution" default the way '
    'there is for target_n, since this script always conditions on '
    'the property (that\'s the point of this script existing).')
flags.DEFINE_float('property_mean', 2.6520,
                   'Must match the value used at training time.')
flags.DEFINE_float('property_std', 1.4035,
                   'Must match the value used at training time.')
flags.DEFINE_string(
    'save_results_json', '',
    'Same format as generate_molecules_n_conditioned.py, plus a '
    'target_property field in the summary.')

flags.DEFINE_integer('node_embedding_dim', 64, '')

flags.DEFINE_integer('p2_latent_dim', 256, '')
flags.DEFINE_integer('p2_num_mlp_layers', 2, '')
flags.DEFINE_integer('p2_num_processing_steps', 6, '')
flags.DEFINE_integer('p2_attn_kq_dim', 32, '')
flags.DEFINE_integer('p2_attn_v_dim', 32, '')
flags.DEFINE_integer('p2_attn_num_heads', 2, '')
flags.DEFINE_integer('p2_attn_concat_heads_output_dim', 32, '')
flags.DEFINE_bool('p2_weight_sharing', True, '')
flags.DEFINE_integer('p2_num_bond_refine_steps', 1, '')

flags.DEFINE_integer('flow_num_coupling_layers', 12, '')
flags.DEFINE_integer('flow_hidden_dim', 256, '')
flags.DEFINE_integer('flow_n_embed_dim', 32, '')
flags.DEFINE_bool('flow_weight_sharing', False, '')
flags.DEFINE_float('flow_max_log_scale', 0.25, '')

FLAGS = flags.FLAGS


def transform_example(n_node):
    edges, receivers, senders, n_edge = None, None, None, None
    senders, receivers = senders_receivers(n_node)
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge = tf.square(n_node)
    edges = tf.zeros_like(senders)
    return edges, receivers, senders, n_edge


def build_flow_graph():
    """Restores the N+property-conditioned flow into its own Graph/
    Session. Returns (sess, sample_n_node_placeholder,
    sample_property_placeholder, generated_embeddings_tensor).
    """
    graph = tf.Graph()
    with graph.as_default():
        node_embeddings_placeholder = tf.placeholder(
            tf.float32, shape=[None, FLAGS.node_embedding_dim],
            name='node_embeddings_placeholder')
        n_node_placeholder = tf.placeholder(tf.int32, shape=[None],
                                            name='n_node_placeholder')
        property_placeholder = tf.placeholder(tf.float32, shape=[None],
                                              name='property_placeholder')
        edges, receivers, senders, n_edge = transform_example(
            n_node_placeholder)
        graphs_tuple = gn.graphs.GraphsTuple(nodes=node_embeddings_placeholder,
                                             edges=edges,
                                             globals=property_placeholder,
                                             receivers=receivers,
                                             senders=senders,
                                             n_node=n_node_placeholder,
                                             n_edge=n_edge)

        half_dim = FLAGS.node_embedding_dim // 2
        flow_embedding = NPropertyEmbedding(FLAGS.flow_n_embed_dim,
                                           FLAGS.property_mean,
                                           FLAGS.property_std)
        grevnet = NConditionedMolecularGNFBlock(
            num_timesteps=FLAGS.flow_num_coupling_layers,
            node_embedding_dim=half_dim,
            hidden_dim=FLAGS.flow_hidden_dim,
            n_embed_dim=FLAGS.flow_n_embed_dim,
            weight_sharing=FLAGS.flow_weight_sharing,
            max_log_scale=FLAGS.flow_max_log_scale,
            embedding_module=flow_embedding)
        prior_embedding = NPropertyEmbedding(FLAGS.flow_n_embed_dim,
                                            FLAGS.property_mean,
                                            FLAGS.property_std)
        prior_net = ConditionalPriorNetwork(FLAGS.node_embedding_dim)

        # Dummy f() call: never run, only needed so every one of the
        # coupling sub-networks' variables (and the prior network's)
        # get created now, in the same traversal order training used.
        grevnet(graphs_tuple, inverse=True)
        dummy_n_embedding = prior_embedding(graphs_tuple)
        prior_net(dummy_n_embedding)

        sample_n_node_placeholder = tf.placeholder(
            tf.int32, shape=[None], name='sample_n_node_placeholder')
        sample_property_placeholder = tf.placeholder(
            tf.float32, shape=[None], name='sample_property_placeholder')
        s_edges, s_receivers, s_senders, s_n_edge = transform_example(
            sample_n_node_placeholder)
        sample_graph_phs = gn.graphs.GraphsTuple(
            nodes=tf.zeros([tf.reduce_sum(sample_n_node_placeholder),
                           FLAGS.node_embedding_dim]),
            edges=s_edges, globals=sample_property_placeholder,
            receivers=s_receivers, senders=s_senders,
            n_node=sample_n_node_placeholder, n_edge=s_n_edge)

        sample_n_embedding = prior_embedding(sample_graph_phs)
        mu, sigma = prior_net(sample_n_embedding)
        sampled_nodes = sample_conditional_prior(sample_n_node_placeholder,
                                                 mu, sigma)
        sample_graph_phs = sample_graph_phs.replace(nodes=sampled_nodes)

        generated = grevnet(sample_graph_phs, inverse=False)

        sess = reset_sess()
        saver = tf.train.Saver()
        print("Restoring N+property-conditioned flow from {}".format(
            FLAGS.flow_checkpoint))
        saver.restore(sess, FLAGS.flow_checkpoint)

    return (sess, sample_n_node_placeholder, sample_property_placeholder,
            generated.nodes)


def build_decoder_graph():
    """Restores Phase 2's frozen encoder/decoder -- unmodified copy of
    generate_molecules_n_conditioned.py's version.
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
    if FLAGS.target_property != FLAGS.target_property:  # NaN check
        raise ValueError("--target_property is required (e.g. --target_property=3.0 "
                         "for a molecule with dipole moment ~3.0 Debye).")
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
    print("--target_property={} (target_n={})".format(
        FLAGS.target_property,
        FLAGS.target_n if FLAGS.target_n > 0 else "sampled"))

    flow_sess, sample_n_node_ph, sample_property_ph, generated_embeddings_t = build_flow_graph()
    decoder_sess, embeddings_ph, atom_pred_t, bond_probs_t = build_decoder_graph()

    results = []
    num_generated = 0
    while num_generated < FLAGS.num_molecules_to_generate:
        batch_size = min(FLAGS.sample_batch_size,
                         FLAGS.num_molecules_to_generate - num_generated)
        if FLAGS.target_n > 0:
            batch_n_node = [FLAGS.target_n] * batch_size
        else:
            batch_n_node = [random.choice(train_n_node)
                           for _ in range(batch_size)]
        batch_property = [FLAGS.target_property] * batch_size

        embeddings = flow_sess.run(
            generated_embeddings_t,
            feed_dict={sample_n_node_ph: batch_n_node,
                      sample_property_ph: batch_property})

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

    # NOTE: this does not yet check whether the DECODED molecules
    # actually hit the requested property value -- that's the real
    # calibration question flagged when this design was first
    # discussed (unlike N, a structural guarantee, the property is
    # only a statistical target with no guarantee any individual
    # sample lands near it). Computing achieved dipole moment from a
    # bare SMILES isn't a one-line RDKit descriptor call the way QED
    # is -- it needs a 3D conformer plus a partial-charge method (e.g.
    # AllChem.EmbedMolecule + Gasteiger charges), which is a real,
    # separate follow-up diagnostic, not folded into this script to
    # avoid shipping a wrong or misleading number.

    print("\n" + "=" * 60)
    print("Generated {} molecules (target_n={}, target_property={})".format(
        n, FLAGS.target_n if FLAGS.target_n > 0 else "sampled",
        FLAGS.target_property))
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
    print("Strict validity (valid AND connected): {}/{} ({:.1f}%)".format(
        len(connected), n, 100 * strict_validity))
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
                'target_n': FLAGS.target_n,
                'target_property': FLAGS.target_property,
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
