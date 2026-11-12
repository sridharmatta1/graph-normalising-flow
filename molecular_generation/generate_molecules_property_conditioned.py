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

For a property that RDKit computes from the graph (derived_properties.py:
logp, tpsa) it also scores every strictly valid generated molecule and
reports how close the achieved value is to the requested one, per target
(--target_properties takes a comma-separated list). For QM9's stored
quantum properties (mu, homo, ...) there is nothing to compute from a
graph, so only the usual validity/uniqueness/novelty is reported.

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
from coupling_variants import (attention_kwargs_from_flags,
                               build_n_conditioned_block)
from derived_properties import DERIVED_PROPERTIES, mol_property
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
flags.DEFINE_string(
    'target_properties', '',
    'Comma-separated target values, e.g. "-1.156,0.299,1.948": generates '
    '--num_molecules_to_generate molecules for EACH and reports each target '
    'separately. Overrides --target_property when given.')
flags.DEFINE_string(
    'property_name', 'mu',
    'The conditioning property the flow was trained on. If it is one of '
    'derived_properties.DERIVED_PROPERTIES (logp, tpsa) the achieved value of '
    'every strictly valid molecule is computed and compared with the target.')
flags.DEFINE_float('property_mean', 2.6520,
                   'Must match the value used at training time.')
flags.DEFINE_float('property_std', 1.4035,
                   'Must match the value used at training time.')
flags.DEFINE_float(
    'sigma_scale', 1.0,
    'Multiplies the trained prior sigma at sampling time, exactly as in '
    'generate_molecules_n_conditioned.py. 1.0 = as trained.')
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
flags.DEFINE_enum(
    'coupling_arch', 'film',
    ['film', 'attn_film', 'deep_film', 'interaction_film', 'spline_film'],
    'Must match the architecture the flow was trained with.')
flags.DEFINE_bool('channel_permutation', False,
                  'Must match training.')
flags.DEFINE_integer('spline_num_bins', 8, 'spline_film only.')
flags.DEFINE_float('spline_bound', 5.0, 'spline_film only.')

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

        flow_embedding = NPropertyEmbedding(FLAGS.flow_n_embed_dim,
                                           FLAGS.property_mean,
                                           FLAGS.property_std)
        grevnet = build_n_conditioned_block(
            FLAGS.coupling_arch,
            num_timesteps=FLAGS.flow_num_coupling_layers,
            node_embedding_dim=FLAGS.node_embedding_dim,
            hidden_dim=FLAGS.flow_hidden_dim,
            n_embed_dim=FLAGS.flow_n_embed_dim,
            weight_sharing=FLAGS.flow_weight_sharing,
            max_log_scale=FLAGS.flow_max_log_scale,
            embedding_module=flow_embedding,
            channel_permutation=FLAGS.channel_permutation,
            spline_num_bins=FLAGS.spline_num_bins,
            spline_bound=FLAGS.spline_bound,
            attn_kwargs=attention_kwargs_from_flags())
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
        sigma = sigma * FLAGS.sigma_scale
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


def parse_targets():
    if FLAGS.target_properties:
        return [float(x) for x in FLAGS.target_properties.split(',')]
    if FLAGS.target_property != FLAGS.target_property:  # NaN check
        raise ValueError("--target_property (or --target_properties) is "
                         "required, e.g. --target_property=0.3 for a molecule "
                         "with logP ~0.3.")
    return [FLAGS.target_property]


def summarize(rows, train_smiles, target, can_score):
    """rows: list of (smiles, is_connected, achieved)."""
    n = len(rows)
    valid = [r[0] for r in rows if r[0] is not None]
    connected = [r for r in rows if r[0] is not None and r[1]]
    connected_smiles = [r[0] for r in connected]
    unique = set(connected_smiles)
    novel = [s for s in unique if s not in train_smiles]
    out = {
        'target': target,
        'num_generated': n,
        'validity': len(valid) / n,
        'strict_validity': len(connected) / n,
        'connected_uniqueness': (len(unique) / len(connected)
                                 if connected else 0.0),
        'connected_novelty': len(novel) / len(unique) if unique else 0.0,
        'num_strict_valid': len(connected),
        'num_unique': len(unique),
        'num_novel': len(novel),
    }
    if can_score:
        vals = np.array([r[2] for r in connected if r[2] is not None])
        if len(vals):
            out.update({
                'achieved_mean': float(vals.mean()),
                'achieved_std': float(vals.std()),
                'achieved_median': float(np.median(vals)),
                'mean_abs_error': float(np.abs(vals - target).mean()),
                'bias': float((vals - target).mean()),
                'num_scored': int(len(vals)),
            })
    return out


def main(argv):
    del argv
    targets = parse_targets()
    can_score = FLAGS.property_name in DERIVED_PROPERTIES
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
    print("property={} targets={} (target_n={}) sigma_scale={} "
          "coupling_arch={}".format(
              FLAGS.property_name, targets,
              FLAGS.target_n if FLAGS.target_n > 0 else "sampled",
              FLAGS.sigma_scale, FLAGS.coupling_arch))

    train_reference = None
    if can_score:
        ref_vals = []
        for e in random.sample(train_examples, min(5000, len(train_examples))):
            ref_vals.append(mol_property(
                graph_to_mol(e['atom_types'], e['bond_matrix']),
                FLAGS.property_name))
        ref_vals = np.array(ref_vals)
        train_reference = {'mean': float(ref_vals.mean()),
                           'std': float(ref_vals.std()),
                           'percentiles_5_25_50_75_95': [
                               float(x) for x in
                               np.percentile(ref_vals, [5, 25, 50, 75, 95])]}
        print("Training-set {} (5000 sample): mean {:.3f} std {:.3f}".format(
            FLAGS.property_name, train_reference['mean'],
            train_reference['std']))

    flow_sess, sample_n_node_ph, sample_property_ph, generated_embeddings_t = build_flow_graph()
    decoder_sess, embeddings_ph, atom_pred_t, bond_probs_t = build_decoder_graph()

    per_target, all_rows = [], []
    for target in targets:
        rows = []
        num_generated = 0
        while num_generated < FLAGS.num_molecules_to_generate:
            batch_size = min(FLAGS.sample_batch_size,
                             FLAGS.num_molecules_to_generate - num_generated)
            if FLAGS.target_n > 0:
                batch_n_node = [FLAGS.target_n] * batch_size
            else:
                batch_n_node = [random.choice(train_n_node)
                               for _ in range(batch_size)]
            batch_property = [target] * batch_size

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
                achieved = None
                try:
                    mol = graph_to_mol(mol_atom_pred, bond_matrix)
                    smiles = Chem.MolToSmiles(mol)
                    is_connected = len(Chem.GetMolFrags(mol)) == 1
                    if can_score and is_connected:
                        achieved = mol_property(mol, FLAGS.property_name)
                except Exception:
                    smiles = None
                    is_connected = None
                rows.append((smiles, is_connected, achieved))

            num_generated += batch_size
            print("target {}: generated {}/{} molecules".format(
                target, num_generated, FLAGS.num_molecules_to_generate))

        summary = summarize(rows, train_smiles, target, can_score)
        per_target.append(summary)
        all_rows.append((target, rows))

        # Lines keep the exact wording generate_molecules_n_conditioned.py
        # prints, so the same grep patterns pick the metrics out of the log.
        print("\n" + "=" * 60)
        print("target_property={} sigma_scale={}".format(target,
                                                         FLAGS.sigma_scale))
        print("Strict validity (valid AND connected -- a real single "
              "molecule): {}/{} ({:.1f}%)".format(
                  summary['num_strict_valid'], summary['num_generated'],
                  100 * summary['strict_validity']))
        print("  Uniqueness (of strictly valid): {}/{} ({:.1f}%)".format(
            summary['num_unique'], summary['num_strict_valid'],
            100 * summary['connected_uniqueness']))
        print("  Novelty (of unique strictly-valid, vs. training set): "
              "{}/{} ({:.1f}%)".format(
                  summary['num_novel'], summary['num_unique'],
                  100 * summary['connected_novelty']))
        if 'achieved_mean' in summary:
            print("  Achieved {} (strictly valid, n={}): mean {:.3f} std {:.3f} "
                  "| target {:.3f} bias {:+.3f} MAE {:.3f}".format(
                      FLAGS.property_name, summary['num_scored'],
                      summary['achieved_mean'], summary['achieved_std'],
                      target, summary['bias'], summary['mean_abs_error']))
        print("=" * 60)

    overall = {}
    if can_score:
        xs = np.array([t['target'] for t in per_target
                       if 'achieved_mean' in t])
        ys = np.array([t['achieved_mean'] for t in per_target
                       if 'achieved_mean' in t])
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            overall = {'slope_of_achieved_mean_vs_target': float(slope),
                       'intercept': float(intercept),
                       'corr_of_target_and_achieved_mean':
                           float(np.corrcoef(xs, ys)[0, 1])}
            print("\nAchieved mean vs target across {} targets: slope {:.3f} "
                  "(1.0 = follows the request exactly, 0 = ignores it), "
                  "correlation {:.3f}".format(
                      len(xs), overall['slope_of_achieved_mean_vs_target'],
                      overall['corr_of_target_and_achieved_mean']))

    if FLAGS.save_results_json:
        molecules = []
        for target, rows in all_rows:
            for smiles, is_connected, achieved in rows:
                molecules.append({
                    'target': target,
                    'smiles': smiles,
                    'connected': bool(is_connected) if is_connected is not None else False,
                    'novel': (smiles not in train_smiles) if
                             (smiles is not None and is_connected) else False,
                    'achieved': achieved,
                })
        payload = {
            'config': {
                'property_name': FLAGS.property_name,
                'sigma_scale': FLAGS.sigma_scale,
                'coupling_arch': FLAGS.coupling_arch,
                'target_n': FLAGS.target_n,
                'num_molecules_per_target': FLAGS.num_molecules_to_generate,
                'flow_checkpoint': FLAGS.flow_checkpoint,
            },
            'train_reference': train_reference,
            'per_target': per_target,
            'overall': overall,
            'molecules': molecules,
        }
        os.makedirs(os.path.dirname(FLAGS.save_results_json) or '.',
                   exist_ok=True)
        with open(FLAGS.save_results_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print("\nSaved {} generated molecules + per-target summaries to {}".format(
            len(molecules), FLAGS.save_results_json))


if __name__ == '__main__':
    app.run(main)
