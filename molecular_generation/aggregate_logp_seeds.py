"""Aggregates the logP-conditioned generation results over seeds.

Reads the generated_logp_A*_full_sigma<scale>.json files written by
run_molecular_flow_logp.sh (the seed-12345 run has no suffix, the others end in
_s<seed>) and prints, per requested logP target, the achieved logP, the share of
valid molecules within +-0.25 / +-0.5 of the target and the usual strict
validity / uniqueness / novelty, as mean +- sd over seeds; plus one row per
seed (slope and correlation of achieved mean vs target, overall N.U.V.).

--reference_json (a generate_molecules_n_conditioned.py result, e.g. plain A's
generated_molecules_arch_A_full_sigma1.0_10000.json) adds the hit rates of a
model that was never told a target, computed with RDKit from its SMILES.

No TensorFlow needed; RDKit only when --reference_json is given.
"""

import argparse
import glob
import json
import os
import re

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    by = {}
    for m in d['molecules']:
        if m['connected'] and m['achieved'] is not None:
            by.setdefault(m['target'], []).append(m['achieved'])
    return d, {t: np.array(v) for t, v in by.items()}


def hit(values, target, tol):
    return 100.0 * float(np.mean(np.abs(values - target) <= tol))


def pm(xs, digits=1):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return "{:.{d}f}".format(xs[0], d=digits)
    return "{:.{d}f} +/- {:.{d}f}".format(xs.mean(), xs.std(ddof=1), d=digits)


def seed_of(path, tag):
    m = re.match(r'generated_' + re.escape(tag) + r'(?:_s(\d+))?_full_', os.path.basename(path))
    return int(m.group(1)) if m and m.group(1) else 12345


def reference_values(path):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen
    RDLogger.DisableLog('rdApp.*')
    with open(path) as f:
        d = json.load(f)
    return np.array([Crippen.MolLogP(Chem.MolFromSmiles(m['smiles']))
                     for m in d['molecules'] if m['connected'] and m['smiles']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='molecular_generation/results')
    p.add_argument('--tag', default='logp_A', help='run tag prefix (logp_A, logp_NZ0.15, ...)')
    p.add_argument('--scale', default='1.0')
    p.add_argument('--reference_json', default='')
    p.add_argument('--save_json', default='')
    args = p.parse_args()

    pattern = os.path.join(args.results_dir, 'generated_{}*_full_sigma{}.json'.format(args.tag, args.scale))
    files = sorted(f for f in glob.glob(pattern)
                   if re.match(r'generated_' + re.escape(args.tag) + r'(?:_s\d+)?_full_', os.path.basename(f)))
    if not files:
        raise SystemExit("no files match {}".format(pattern))
    runs = {seed_of(f, args.tag): load(f) for f in files}
    seeds = sorted(runs)
    targets = sorted(next(iter(runs.values()))[1])
    ref = reference_values(args.reference_json) if args.reference_json else None
    print("{} seeds: {} (sigma scale {})".format(len(seeds), seeds, args.scale))

    per_seed = []
    print("\nPer seed")
    print("seed   slope   corr   N.U.V.   strict valid")
    for s in seeds:
        d, by = runs[s]
        means = [by[t].mean() for t in targets]
        slope = float(np.polyfit(targets, means, 1)[0])
        corr = float(np.corrcoef(targets, means)[0, 1])
        pt = d['per_target']
        total = sum(x['num_generated'] for x in pt)
        nuv = 100.0 * sum(x['num_novel'] for x in pt) / total
        valid = 100.0 * sum(x['num_strict_valid'] for x in pt) / total
        per_seed.append(dict(seed=s, slope=slope, corr=corr, nuv=nuv, strict_validity=valid))
        print("{:<6} {:5.3f}  {:5.3f}  {:6.2f}   {:6.2f}".format(s, slope, corr, nuv, valid))
    print("mean   {}  {}  {}   {}".format(
        pm([x['slope'] for x in per_seed], 3), pm([x['corr'] for x in per_seed], 3),
        pm([x['nuv'] for x in per_seed], 2), pm([x['strict_validity'] for x in per_seed], 2)))

    print("\nPer target (mean +/- sd over seeds)")
    header = "target | achieved mean | within 0.25% | within 0.5% | strict valid | unique | novel"
    if ref is not None:
        header += " | plain A within 0.25 / 0.5"
    print(header)
    rows = []
    for t in targets:
        ach = [runs[s][1][t].mean() for s in seeds]
        h25 = [hit(runs[s][1][t], t, 0.25) for s in seeds]
        h50 = [hit(runs[s][1][t], t, 0.5) for s in seeds]
        pts = [next(x for x in runs[s][0]['per_target'] if x['target'] == t) for s in seeds]
        row = dict(target=t, achieved=pm(ach, 3), within_025=pm(h25), within_05=pm(h50),
                   strict_validity=pm([100 * x['strict_validity'] for x in pts]),
                   uniqueness=pm([100 * x['connected_uniqueness'] for x in pts]),
                   novelty=pm([100 * x['connected_novelty'] for x in pts]))
        line = "{:7.3f} | {} | {} | {} | {} | {} | {}".format(
            t, row['achieved'], row['within_025'], row['within_05'], row['strict_validity'],
            row['uniqueness'], row['novelty'])
        if ref is not None:
            row['plain_within_025'], row['plain_within_05'] = hit(ref, t, 0.25), hit(ref, t, 0.5)
            line += " | {:.1f} / {:.1f}".format(row['plain_within_025'], row['plain_within_05'])
        rows.append(row)
        print(line)
    if ref is not None:
        print("plain A reference: n {} mean {:.3f} std {:.3f}".format(len(ref), ref.mean(), ref.std()))

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump({'seeds': seeds, 'scale': args.scale, 'per_seed': per_seed, 'per_target': rows}, f, indent=2)
        print("\nSaved {}".format(args.save_json))


if __name__ == '__main__':
    main()
