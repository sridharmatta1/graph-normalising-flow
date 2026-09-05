"""Shared RDKit conversion logic between SMILES and the (atom_types,
bond_matrix) graph representation, kept separate from preprocess_qm9.py
for the same reason qm9_constants.py is: preprocess_qm9.py's
absl.flags.DEFINE_* calls run at import time and collide with any other
script in the same process that defines a flag of the same name (bit
this project has already been hit by once for ATOM_VOCAB -- this file
is the same fix, extended to graph_to_mol/mol_to_graph so future
scripts, like a molecular-autoencoder round-trip checker, can reuse
them without pulling in preprocess_qm9.py's flags too).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

from qm9_constants import ATOM_VOCAB, ATOM_TO_IDX

RDLogger.DisableLog('rdApp.*')

# 0 = no bond. Aromatic bonds are resolved via Kekulization before this
# mapping is applied, so RDKit should never hand us BondType.AROMATIC.
BOND_TO_IDX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
IDX_TO_BOND = {v: k for k, v in BOND_TO_IDX.items()}

# Standard valence (max total bond order) for a neutral atom of each
# element -- matches mol_to_graph's assumption that QM9 molecules are
# always neutral (formal charge 0).
VALENCE = {'C': 4, 'N': 3, 'O': 2, 'F': 1}
ATOM_VALENCE = [VALENCE[a] for a in ATOM_VOCAB]


def decode_bonds_valence_aware(atom_types, bond_probs):
    """Greedily builds a valence-respecting bond_matrix from per-pair bond
    class probabilities, instead of independently argmax-ing every pair.

    Independent per-pair argmax (what generate_graphs.py-style decoding
    would naively do) has no way to know that two OTHER predicted bonds
    on the same atom already used up its valence budget -- each pair is
    scored in isolation. This is exactly why ~15% of round-trip
    reconstructions came back invalid (an atom ending up with too many
    bonds) despite ~97% per-pair accuracy: a small number of independently
    "confident" predictions can still collectively overshoot an atom's
    valence.

    This processes candidate bonds in descending order of predicted-class
    confidence, accepting each only if it fits within both endpoint
    atoms' remaining valence budget -- downgrading to a lower bond order
    (e.g. double -> single) rather than rejecting outright when the full
    predicted order doesn't fit, to preserve as much of the model's
    belief as the hard constraint allows. The result is valid by
    construction: no atom can ever exceed its valence.

    atom_types: sequence of length n, values indexing ATOM_VOCAB.
    bond_probs: [n, n, NUM_BOND_TYPES] array of per-pair class
        probabilities (e.g. softmax of the model's bond logits).
    """
    n = len(atom_types)
    remaining_valence = np.array(
        [ATOM_VALENCE[a] for a in atom_types], dtype=np.int32)
    bond_matrix = np.zeros((n, n), dtype=np.int32)

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            pred_order = int(np.argmax(bond_probs[i, j]))
            if pred_order == 0:
                continue
            confidence = float(bond_probs[i, j, pred_order])
            candidates.append((confidence, i, j, pred_order))
    candidates.sort(key=lambda c: c[0], reverse=True)

    for confidence, i, j, pred_order in candidates:
        order = min(pred_order, remaining_valence[i], remaining_valence[j])
        if order >= 1:
            bond_matrix[i, j] = order
            bond_matrix[j, i] = order
            remaining_valence[i] -= order
            remaining_valence[j] -= order

    return bond_matrix


def mol_to_graph(smiles):
    """SMILES -> (atom_types, bond_matrix), or None if unsupported."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Chem.KekulizeException:
        return None

    atom_types = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_TO_IDX:
            return None  # outside the QM9 vocab -- shouldn't happen.
        if atom.GetFormalCharge() != 0:
            return None  # QM9 molecules are neutral; be defensive.
        atom_types.append(ATOM_TO_IDX[symbol])

    n = mol.GetNumAtoms()
    bond_matrix = np.zeros((n, n), dtype=np.int32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if bond_type not in BOND_TO_IDX:
            return None  # e.g. unexpected bond type after Kekulize.
        bond_matrix[i, j] = BOND_TO_IDX[bond_type]
        bond_matrix[j, i] = BOND_TO_IDX[bond_type]

    return atom_types, bond_matrix


def graph_to_mol(atom_types, bond_matrix):
    """Inverse of mol_to_graph. Raises if the atom/bond combination isn't
    a valid molecule (e.g. an atom with too many bonds) -- callers that
    expect this (like decoding a freshly-trained/generated prediction,
    as opposed to Phase 1's own known-good data) should catch exceptions.
    """
    rw_mol = Chem.RWMol()
    for a in atom_types:
        rw_mol.AddAtom(Chem.Atom(ATOM_VOCAB[a]))
    n = len(atom_types)
    for i in range(n):
        for j in range(i + 1, n):
            b = bond_matrix[i, j]
            if b != 0:
                rw_mol.AddBond(i, j, IDX_TO_BOND[b])
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol
