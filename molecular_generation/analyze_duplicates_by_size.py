"""One-off diagnostic (not part of the Phase 1-6 pipeline): checks
whether the n=10,000 generation run's uniqueness shortfall (93.3%
overall, vs MoFlow's 99.2%) is concentrated in small molecules, rather
than being uniform across all sizes.

Motivation: N-conditioning (Phase 4) is still deferred, so Phase 6
draws each generated molecule's heavy-atom count N from QM9's own
training-set N distribution. QM9's own chemical space for small N
(e.g. N=1-3) is inherently tiny -- there are only a handful of
chemically distinct molecules expressible from {C,N,O,F} with 1-3
heavy atoms at all, independent of how good the generator is. If small
N draws are the ones colliding, the uniqueness shortfall is largely a
sampling/combinatorial artifact of not yet conditioning on N, not
necessarily a flow-diversity problem the earlier temperature sweep
would ever have been able to fix. If duplicates are spread evenly
across all sizes instead, that points the other way -- toward the flow
itself under-covering its embedding space regardless of size.

Only needs the already-saved generation JSON (generate_molecules.py
--save_results_json) and the QM9 training pickle -- no checkpoints, no
TF, no GPU. Heavy-atom count per generated molecule is read back out
of its own canonical SMILES via RDKit rather than needing
generate_molecules.py to have saved n_node separately.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json
import pickle

from absl import app
from absl import flags
from rdkit import Chem

flags.DEFINE_string(
    'results_json',
    'molecular_generation/results/generated_molecules_10k.json', '')
flags.DEFINE_string('train_data_path',
                    'molecular_generation/data/qm9_train.p', '')
FLAGS = flags.FLAGS


def main(argv):
    del argv
    with open(FLAGS.results_json, 'r') as f:
        payload = json.load(f)

    generated_by_size = defaultdict(list)
    for m in payload['molecules']:
        if not m['connected'] or not m['smiles']:
            continue
        mol = Chem.MolFromSmiles(m['smiles'])
        if mol is None:
            continue
        generated_by_size[mol.GetNumAtoms()].append(m['smiles'])

    with open(FLAGS.train_data_path, 'rb') as f:
        train_examples = pickle.load(f)
    train_smiles_by_size = defaultdict(set)
    train_count_by_size = defaultdict(int)
    for e in train_examples:
        train_smiles_by_size[e['n_node']].add(
            Chem.MolToSmiles(Chem.MolFromSmiles(e['smiles'])))
        train_count_by_size[e['n_node']] += 1

    total_generated = sum(len(v) for v in generated_by_size.values())
    print("{:>3} | {:>10} {:>8} {:>10} | {:>14} | {:>14} {:>12}".format(
        "N", "generated", "unique", "uniq_pct", "share_of_total",
        "train_distinct", "train_total"))
    print("-" * 90)
    for n in sorted(generated_by_size):
        smiles_list = generated_by_size[n]
        unique_count = len(set(smiles_list))
        print("{:>3} | {:>10} {:>8} {:>9.1f}% | {:>13.1f}% | {:>14} {:>12}".format(
            n, len(smiles_list), unique_count,
            100.0 * unique_count / len(smiles_list),
            100.0 * len(smiles_list) / total_generated,
            len(train_smiles_by_size.get(n, [])),
            train_count_by_size.get(n, 0)))


if __name__ == '__main__':
    app.run(main)
