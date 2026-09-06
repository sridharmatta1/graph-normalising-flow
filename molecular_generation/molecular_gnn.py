"""Phase 2, part 2: the molecular encoder-decoder heads and losses that
replace gnn.py's single edge/no-edge decoder (pred_adj / binary_loss)
with atom-type and bond-type classification -- needed because molecules
have real per-node atom identity and real per-edge bond-type categories
that community/ego's anonymous structural graphs never had.

Reuses gnn.py's TimestepGNN (the actual message-passing machinery) and
make_mlp_model completely unchanged. Only new here: the input embedding
(one-hot atom features -> node_embedding_dim) and the two decoder heads
(atom-type, bond-type). TimestepGNN's residual connection needs its input
and every processing step's output to already be node_embedding_dim-sized
throughout -- community/ego's random-noise node features were already
that size, so the original code never needed a separate embedding step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import tensorflow as tf

from gnn import make_mlp_model
from loss import loss_mask, remove_diag
from qm9_chem import ATOM_VALENCE
from qm9_graph_data import NUM_ATOM_TYPES, NUM_BOND_TYPES

BOND_ORDER = [0.0, 1.0, 2.0, 3.0]  # bond class index -> valence it consumes.


def embed_atom_features(graph_phs, latent_dim, node_embedding_dim,
                        num_layers=2):
    """Raw one-hot atom features (dim=NUM_ATOM_TYPES) -> node_embedding_dim,
    so the result can be fed into TimestepGNN. Call this once on the raw
    placeholder before running TimestepGNN; keep the raw graph_phs around
    separately too, since true_atom_type() needs its original one-hot
    values, not the projected embedding.
    """
    embed_mlp = make_mlp_model(latent_dim, node_embedding_dim, num_layers)
    return graph_phs.replace(nodes=embed_mlp(graph_phs.nodes))


def atom_type_logits(gnn_output, latent_dim, num_layers=2):
    head = make_mlp_model(latent_dim, NUM_ATOM_TYPES, num_layers)
    return head(gnn_output.nodes)  # [N_total, NUM_ATOM_TYPES]


def bond_type_logits(gnn_output, node_embedding_dim, latent_dim,
                     num_layers=2):
    """Pairwise, symmetric function of every (i, j) node-embedding pair.
    Symmetric combination (sum / abs-diff / product) so bond(i, j) and
    bond(j, i) get identical logits -- a bond has no direction, unlike
    the original decoder's directed adjacency (which didn't matter there
    since it was reconstructing an undirected structural graph anyway,
    just never needed to be explicitly enforced).
    """
    x = gnn_output.nodes  # [N, D]
    n = tf.shape(x)[0]
    xi = tf.tile(tf.expand_dims(x, 1), [1, n, 1])  # [N, N, D]
    xj = tf.tile(tf.expand_dims(x, 0), [n, 1, 1])  # [N, N, D]
    pair_features = tf.concat([xi + xj, tf.abs(xi - xj), xi * xj], axis=-1)
    flat = tf.reshape(pair_features, [-1, 3 * node_embedding_dim])
    head = make_mlp_model(latent_dim, NUM_BOND_TYPES, num_layers)
    logits_flat = head(flat)  # [N*N, NUM_BOND_TYPES]
    return tf.reshape(logits_flat, [n, n, NUM_BOND_TYPES])


def refine_bond_logits(gnn_output, raw_graph_phs, atom_labels,
                       node_embedding_dim, latent_dim, num_layers,
                       num_refine_steps=1):
    """Iteratively refines bond-type logits over num_refine_steps rounds,
    each conditioned on the current round's *expected* remaining valence
    per atom (computed softly/differentiably from the current bond
    probabilities) -- lets bond decisions on the same atom inform each
    other, instead of a single independent shot from node embeddings
    alone (what bond_type_logits does on its own).

    This targets the "right atom/bond count, wrong specific position"
    errors left over after class-weighting and more model capacity
    stopped helping (see run_molecular_autoencoder_v2/v3.sh) -- e.g. a
    double bond predicted one position over from where it actually is.
    Neither of those fixes lets one bond decision influence another;
    this does, by feeding each round's tentative global bond assignment
    back into the node embeddings before re-predicting.

    A full sequential/autoregressive decoder (fixed left-to-right order,
    each pair conditioned strictly on already-decided earlier pairs) was
    considered instead, but needs a tf.while_loop with a growing,
    gather/scatter-heavy state -- too easy to get subtly wrong to write
    blind (this project's dev machine can't run a real TF graph to test
    it before it reaches the cluster). This version reuses the static,
    fixed-number-of-rounds unrolling loss_mask() already uses elsewhere
    in this codebase -- every round has the same shapes, so there's much
    less room for a shape/index bug to hide in.

    num_refine_steps=1 (the default) reproduces the original single-shot
    behavior exactly (the loop body never runs).

    atom_labels must be the ground-truth atom type per node (this
    reconstructs real molecules the model was shown, not free
    generation -- Phase 6 would need to decide how to get atom types
    for genuinely novel samples, a separate problem).
    """
    mask = remove_diag(loss_mask(raw_graph_phs))
    bond_order = tf.constant(BOND_ORDER, dtype=tf.float32)
    atom_valence = tf.gather(
        tf.constant(ATOM_VALENCE, dtype=tf.float32), atom_labels)  # [N]

    nodes = gnn_output.nodes
    bond_logits = bond_type_logits(gnn_output, node_embedding_dim,
                                   latent_dim, num_layers)

    for _ in range(num_refine_steps - 1):
        bond_probs = tf.nn.softmax(bond_logits, axis=-1)
        expected_order = tf.reduce_sum(bond_probs * bond_order, axis=-1)
        expected_used_valence = tf.reduce_sum(expected_order * mask, axis=1)
        # reshape (not expand_dims) -- loss_mask() builds `mask` via a
        # tf.while_loop with infer_shape=False, so mask's static shape
        # is fully undetermined, and that uncertainty otherwise
        # propagates through the ops above into remaining_valence's
        # shape. reshape with an explicit target shape overrides that
        # rather than inheriting it, which the update_mlp's Linear
        # layer below needs (Sonnet requires a statically-known input
        # size to build its weights).
        remaining_valence = tf.reshape(
            atom_valence - expected_used_valence, [-1, 1])  # [N, 1]

        update_mlp = make_mlp_model(latent_dim, node_embedding_dim,
                                    num_layers)
        update_input = tf.concat([nodes, remaining_valence], axis=1)
        update_input = tf.reshape(
            update_input, [-1, node_embedding_dim + 1])
        nodes = nodes + update_mlp(update_input)

        bond_logits = bond_type_logits(
            gnn_output.replace(nodes=nodes), node_embedding_dim,
            latent_dim, num_layers)

    return bond_logits


def true_atom_type(raw_graph_phs):
    """raw_graph_phs must be the placeholder BEFORE embed_atom_features
    projects it -- its .nodes are still the original one-hot atom labels.
    """
    return tf.argmax(raw_graph_phs.nodes, axis=1, output_type=tf.int32)


def true_bond_type_matrix(raw_graph_phs, dim):
    """Scatters each real bond's type into a dense [dim, dim] matrix --
    analogous to loss.py's adjacency_matrix(), but carrying the actual
    bond-type value (0-3) instead of just edge presence. Self-loop edges
    (feature 0.0) scatter zeros onto the diagonal, which is harmless: the
    diagonal is always masked out of the loss (see
    molecular_reconstruction_loss), same as remove_diag() does for the
    structural-graph binary adjacency loss.
    """
    indices = tf.stack([raw_graph_phs.senders, raw_graph_phs.receivers],
                       axis=1)
    values = tf.cast(tf.reshape(raw_graph_phs.edges, [-1]), tf.int32)
    return tf.scatter_nd(indices, values, [dim, dim])


def molecular_reconstruction_loss(raw_graph_phs, gnn_output,
                                  node_embedding_dim, latent_dim,
                                  num_layers=2, bond_class_weight=1.0,
                                  num_bond_refine_steps=1):
    """Cross-entropy on atom-type + bond-type reconstruction, replacing
    gnn.py's binary_loss (which only ever reconstructed edge presence).
    Masking follows binary_loss's pattern exactly: loss_mask() zeroes out
    cross-graph pairs within a batch (block-diagonal), remove_diag()
    zeroes out self-pairs (meaningless for bonds, same as for edges).

    bond_class_weight (>=1.0) upweights the three real-bond classes
    (single/double/triple) relative to the dominant "no bond" class in
    the loss. Most atom pairs in a molecule aren't bonded, so the
    unweighted loss can hit high aggregate bond_accuracy mostly by
    nailing the easy majority class while still getting real bonds
    wrong -- exactly what shows up as "valid-but-different"
    reconstructions (right shape, wrong bond somewhere) rather than
    exact matches. Weighting the loss towards real bonds specifically
    targets that gap. 1.0 = no reweighting (every pair equal, as before).

    num_bond_refine_steps (>=1) controls refine_bond_logits' iterative
    refinement rounds. 1 = single-shot (original behavior).
    """
    num_nodes = tf.reduce_sum(raw_graph_phs.n_node)

    atom_logits = atom_type_logits(gnn_output, latent_dim, num_layers)
    atom_labels = true_atom_type(raw_graph_phs)
    atom_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=atom_labels, logits=atom_logits)
    atom_loss = tf.reduce_sum(atom_ce)
    atom_pred = tf.argmax(atom_logits, axis=1, output_type=tf.int32)
    atom_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(atom_pred, atom_labels), tf.float32))

    true_bonds = true_bond_type_matrix(raw_graph_phs, num_nodes)
    bond_logits = refine_bond_logits(
        gnn_output, raw_graph_phs, atom_labels, node_embedding_dim,
        latent_dim, num_layers, num_refine_steps=num_bond_refine_steps)
    bond_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_bonds, logits=bond_logits)
    mask = remove_diag(loss_mask(raw_graph_phs))
    class_weight = tf.constant(
        [1.0, bond_class_weight, bond_class_weight, bond_class_weight],
        dtype=tf.float32)
    per_pair_weight = tf.gather(class_weight, true_bonds)
    masked_bond_ce = mask * per_pair_weight * bond_ce
    bond_loss = tf.reduce_sum(masked_bond_ce)
    bond_pred = tf.argmax(bond_logits, axis=2, output_type=tf.int32)
    bond_probs = tf.nn.softmax(bond_logits, axis=-1)
    bond_correct = tf.cast(tf.equal(bond_pred, true_bonds),
                           tf.float32) * mask
    bond_accuracy = tf.reduce_sum(bond_correct) / tf.reduce_sum(mask)

    real_bond_mask = mask * tf.cast(tf.greater(true_bonds, 0), tf.float32)
    real_bond_accuracy = (tf.reduce_sum(bond_correct * real_bond_mask) /
                          tf.maximum(tf.reduce_sum(real_bond_mask), 1.0))

    total_loss = atom_loss + bond_loss
    mean_loss = total_loss / tf.cast(num_nodes, tf.float32)
    return {
        'total_loss': total_loss,
        'mean_loss': mean_loss,
        'atom_loss': atom_loss,
        'bond_loss': bond_loss,
        'atom_accuracy': atom_accuracy,
        'bond_accuracy': bond_accuracy,
        'real_bond_accuracy': real_bond_accuracy,
        'true_bonds': true_bonds,
        'bond_pred': bond_pred,
        'bond_probs': bond_probs,
        'atom_pred': atom_pred,
        'atom_labels': atom_labels,
    }
