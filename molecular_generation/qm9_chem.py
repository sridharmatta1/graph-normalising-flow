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
