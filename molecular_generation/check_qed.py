"""One-off diagnostic (not part of the Phase 1-6 pipeline): computes
RDKit's QED (Quantitative Estimate of Drug-likeness) for every strictly-
valid molecule in a saved generation JSON, and compares against the
same statistic computed over real QM9 training molecules.

QED is a composite score in [0, 1] combining molecular weight, logP,
H-bond donor/acceptor counts, polar surface area, rotatable bonds,
aromatic rings, and structural alerts into one drug-likeness number.
Purely descriptive here -- nothing in this project's training or
generation currently optimizes for it. This is a first look at where
our generated (and real QM9) molecules land on that scale, ahead of
any decision to condition on it (see the discussion on extending
Phase 4 to a property like QED or HOMO-LUMO gap).

QM9 molecules are small (<=9 heavy atoms) and were not selected for
drug-likeness at all, so a low or unremarkable QED here is expected --
this is diagnostic context, not a quality judgment on the generative
model itself.

Only needs the already-saved generation JSON (generate_molecules.py /
generate_molecules_n_conditioned.py --save_results_json) and the QM9
training pickle -- no checkpoints, no TF, no GPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle
import random

from absl import app
from absl import flags
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

flags.DEFINE_string(
    'results_json',
    'molecular_generation/results/generated_molecules_n_conditioned_10k.json',
    '')
flags.DEFINE_string('train_data_path',
                    'molecular_generation/data/qm9_train.p', '')
flags.DEFINE_integer(
    'train_sample_size', 2000,
    'How many real training molecules to sample for the comparison '
    'baseline -- the full training set (105k+) is unnecessary for a '
    'stable mean/std estimate.')
flags.DEFINE_string(
    'save_summary_json', '',
    'If non-empty, also write summary stats (mean/median/std/min/max '
    'for generated and training QED, plus the full generated QED list) '
    'to this path as JSON.')
FLAGS = flags.FLAGS


def qed_stats(values):
    values = np.array(values, dtype=np.float64)
    return {
        'n': len(values),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def main(argv):
    del argv
    with open(FLAGS.results_json, 'r') as f:
        payload = json.load(f)

    generated_qed = []
    generated_qed_novel = []
    generated_qed_seen = []
    for m in payload['molecules']:
        if not m.get('connected') or not m.get('smiles'):
            continue
        mol = Chem.MolFromSmiles(m['smiles'])
        if mol is None:
            continue
        try:
            q = QED.qed(mol)
        except Exception:
            continue
        generated_qed.append(q)
        if m.get('novel'):
            generated_qed_novel.append(q)
        else:
            generated_qed_seen.append(q)

    with open(FLAGS.train_data_path, 'rb') as f:
        train_examples = pickle.load(f)
    sample = random.sample(train_examples,
                           min(FLAGS.train_sample_size, len(train_examples)))
    train_qed = []
    for e in sample:
        mol = Chem.MolFromSmiles(e['smiles'])
        if mol is None:
            continue
        try:
            train_qed.append(QED.qed(mol))
        except Exception:
            continue

    print("Results JSON: {}".format(FLAGS.results_json))
    print("Generated (strictly valid, QED computed): {}".format(
        len(generated_qed)))
    print()

    def report(name, values):
        if not values:
            print("{}: no molecules".format(name))
            return
        s = qed_stats(values)
        print("{:<28} n={:<6} mean={:.4f}  median={:.4f}  std={:.4f}  "
             "min={:.4f}  max={:.4f}".format(
                 name, s['n'], s['mean'], s['median'], s['std'], s['min'],
                 s['max']))

    report("Generated (all)", generated_qed)
    report("Generated (novel)", generated_qed_novel)
    report("Generated (seen in training)", generated_qed_seen)
    report("Real QM9 training sample", train_qed)

    if FLAGS.save_summary_json:
        summary = {
            'results_json': FLAGS.results_json,
            'generated_all': qed_stats(generated_qed) if generated_qed else None,
            'generated_novel': qed_stats(generated_qed_novel) if generated_qed_novel else None,
            'generated_seen': qed_stats(generated_qed_seen) if generated_qed_seen else None,
            'train_sample': qed_stats(train_qed) if train_qed else None,
            'generated_qed_values': generated_qed,
            'train_qed_values': train_qed,
        }
        with open(FLAGS.save_summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print("\nSaved QED summary (incl. raw values, for plotting) to {}".format(
            FLAGS.save_summary_json))


if __name__ == '__main__':
    app.run(main)
