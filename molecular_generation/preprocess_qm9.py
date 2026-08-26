"""Phase 1: QM9 -> atom-type/bond-type graphs.

Downloads (or reads a local copy of) the MoleculeNet QM9 CSV
(SMILES + 12 property columns, ~133885 molecules) and converts each
molecule into the representation the rest of this pipeline needs:
  - atom_types: per-node integer index into a fixed element vocab
  - bond_matrix: NxN integer bond-type matrix (0=none, 1=single,
    2=double, 3=triple -- aromatic rings are Kekulized first so no
    separate "aromatic" class is needed)
  - properties: the raw QM9 property columns, kept for Phase 4
    (property conditioning), unused for now
  - smiles: canonical SMILES, kept only to validate the atom/bond
    representation round-trips losslessly (see --validate)

This is analogous to how community/ego graphs live in
training_graphs/*.dat -- qm9_{train,val,test}.p play the same role
for the molecular pipeline, kept in molecular_generation/data/ so it
never mixes with the existing structure-only pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import os
import pickle
import random
import sys
import urllib.request

from absl import app
from absl import flags
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

# QM9 (v1) molecules only ever contain these 4 heavy elements (plus H,
# which we don't model as explicit nodes -- standard convention for
# graph-based molecule generation, since H count is recoverable from
# valence).
ATOM_VOCAB = ['C', 'N', 'O', 'F']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}

# 0 = no bond. Aromatic bonds are resolved via Kekulization before
# this mapping is applied, so RDKit should never hand us
# BondType.AROMATIC here.
BOND_TO_IDX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
IDX_TO_BOND = {v: k for k, v in BOND_TO_IDX.items()}

flags.DEFINE_string('qm9_csv', '', 'Path to a local qm9.csv. If empty, '
                    'downloads it from QM9_CSV_URL into --output_dir.')
flags.DEFINE_string('output_dir', 'molecular_generation/data', '')
flags.DEFINE_integer(
    'max_molecules', 0,
    'If > 0, only process the first N molecules -- for small-batch '
    'smoke testing before running on the full ~134k dataset.')
flags.DEFINE_float('val_frac', 0.1, '')
flags.DEFINE_float('test_frac', 0.1, '')
flags.DEFINE_integer('random_seed', 12345, '')
flags.DEFINE_bool(
    'validate', True,
    'Reconstruct each molecule from its atom/bond representation and '
    'check the canonical SMILES matches the original -- catches any '
    'lossy encoding before we ever train on this data.')
FLAGS = flags.FLAGS


def download_qm9_csv(dest_path):
    print("Downloading QM9 CSV from {} ...".format(QM9_CSV_URL))
    urllib.request.urlretrieve(QM9_CSV_URL, dest_path)
    print("Saved to {}".format(dest_path))


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
    """Inverse of mol_to_graph -- used only to validate losslessness."""
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


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    csv_path = FLAGS.qm9_csv
    if not csv_path:
        csv_path = os.path.join(FLAGS.output_dir, 'qm9.csv')
        if not os.path.exists(csv_path):
            download_qm9_csv(csv_path)
        else:
            print("Found existing {}, skipping download.".format(csv_path))

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        property_cols = [c for c in reader.fieldnames
                         if c not in ('mol_id', 'smiles')]
        rows = list(reader)

    if FLAGS.max_molecules > 0:
        rows = rows[:FLAGS.max_molecules]
    print("Read {} molecules from {}".format(len(rows), csv_path))

    examples = []
    num_skipped = 0
    num_validate_fail = 0
    n_node_list = []
    atom_vocab_counts = np.zeros(len(ATOM_VOCAB), dtype=np.int64)

    for row in rows:
        smiles = row['smiles']
        graph = mol_to_graph(smiles)
        if graph is None:
            num_skipped += 1
            continue
        atom_types, bond_matrix = graph

        if FLAGS.validate:
            try:
                recon_mol = graph_to_mol(atom_types, bond_matrix)
                recon_smiles = Chem.MolToSmiles(recon_mol)
                orig_canonical = Chem.MolToSmiles(
                    Chem.MolFromSmiles(smiles))
                if recon_smiles != orig_canonical:
                    num_validate_fail += 1
                    continue
            except Exception:
                num_validate_fail += 1
                continue

        properties = {c: float(row[c]) for c in property_cols}
        examples.append({
            'smiles': smiles,
            'atom_types': atom_types,
            'bond_matrix': bond_matrix,
            'n_node': len(atom_types),
            'properties': properties,
        })
        n_node_list.append(len(atom_types))
        for a in atom_types:
            atom_vocab_counts[a] += 1

    print("\n" + "=" * 60)
    print("Processed {} molecules total".format(len(rows)))
    print("  Skipped (parse/kekulize/vocab failure): {}".format(num_skipped))
    if FLAGS.validate:
        print("  Skipped (round-trip validation failure): {}".format(
            num_validate_fail))
    print("  Kept: {}".format(len(examples)))
    print("Atom vocab distribution: {}".format(
        dict(zip(ATOM_VOCAB, atom_vocab_counts.tolist()))))
    print("N (heavy atom count): min={} max={} mean={:.2f}".format(
        min(n_node_list), max(n_node_list), np.mean(n_node_list)))
    print("=" * 60)

    rng = random.Random(FLAGS.random_seed)
    rng.shuffle(examples)
    n = len(examples)
    n_val = int(n * FLAGS.val_frac)
    n_test = int(n * FLAGS.test_frac)
    splits = {
        'val': examples[:n_val],
        'test': examples[n_val:n_val + n_test],
        'train': examples[n_val + n_test:],
    }
    for split_name, split_examples in splits.items():
        out_path = os.path.join(FLAGS.output_dir,
                                'qm9_{}.p'.format(split_name))
        with open(out_path, 'wb') as f:
            pickle.dump(split_examples, f)
        print("Saved {} molecules to {}".format(len(split_examples),
                                                 out_path))


if __name__ == '__main__':
    app.run(main)
