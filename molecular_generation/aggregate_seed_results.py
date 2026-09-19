"""Aggregates generate_molecules.py --save_results_json outputs across
multiple random seeds into a mean +/- std table -- the same reporting
convention MoFlow's own Table 2 uses (mean +/- std over 5 runs),
instead of the single-seed numbers this project has reported so far.

Only reads each seed's saved 'summary' block (validity, connectivity,
strict_validity, valid_uniqueness, valid_novelty, connected_uniqueness,
connected_novelty) -- no checkpoints, no TF, no GPU needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import numpy as np

flags.DEFINE_string(
    'results_json_pattern',
    'molecular_generation/results/generated_molecules_seed_{}_10k.json',
    'Python format string with one {} filled in by each seed number.')
flags.DEFINE_string('seeds', '1,2,3,4,5',
                    'Comma-separated seed numbers to aggregate.')
FLAGS = flags.FLAGS

METRICS = [
    'validity', 'connectivity', 'strict_validity', 'valid_uniqueness',
    'valid_novelty', 'connected_uniqueness', 'connected_novelty',
]


def main(argv):
    del argv
    seeds = [s.strip() for s in FLAGS.seeds.split(',')]

    per_metric = {m: [] for m in METRICS}
    per_seed_summaries = {}
    for seed in seeds:
        path = FLAGS.results_json_pattern.format(seed)
        try:
            with open(path, 'r') as f:
                payload = json.load(f)
        except FileNotFoundError:
            print("WARNING: missing {} -- skipping seed {}".format(path, seed))
            continue
        summary = payload['summary']
        per_seed_summaries[seed] = summary
        for m in METRICS:
            per_metric[m].append(summary[m])

    print("Aggregated over seeds: {}".format(
        list(per_seed_summaries.keys())))
    print("({} of {} requested seeds found)\n".format(
        len(per_seed_summaries), len(seeds)))

    print("{:<24} {:>10} {:>10} {:>10}   per-seed values".format(
        "Metric", "mean", "std", "n_seeds"))
    print("-" * 90)
    for m in METRICS:
        values = per_metric[m]
        if not values:
            print("{:<24} no data".format(m))
            continue
        mean = float(np.mean(values))
        std = float(np.std(values))
        print("{:<24} {:>9.4f}% {:>9.4f}% {:>10}   {}".format(
            m, 100 * mean, 100 * std, len(values),
            ["{:.4f}".format(v) for v in values]))

    # N.U.V. = strict_validity * connected_uniqueness * connected_novelty,
    # per seed (not from the mean of each component, which would be a
    # slightly different -- biased -- quantity).
    nuv_per_seed = []
    for seed, summary in per_seed_summaries.items():
        nuv = (summary['strict_validity'] * summary['connected_uniqueness'] *
              summary['connected_novelty'])
        nuv_per_seed.append(nuv)
    if nuv_per_seed:
        print("-" * 90)
        print("{:<24} {:>9.4f}% {:>9.4f}% {:>10}   {}".format(
            "N.U.V. (implied)", 100 * np.mean(nuv_per_seed),
            100 * np.std(nuv_per_seed), len(nuv_per_seed),
            ["{:.4f}".format(v) for v in nuv_per_seed]))


if __name__ == '__main__':
    app.run(main)
