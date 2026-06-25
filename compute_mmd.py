"""
Compute MMD metrics for generated graphs vs real graphs.
Uses GraphRNN evaluation code (proper EMD kernel + ORCA orbit counting).

Usage:
    python compute_mmd.py \
        --generated_graphs results/ego/seed_1/generated_graphs/graphs.p \
        --dataset graph_rnn_ego_small \
        --output_file results/ego/seed_1/mmd_results.json

Metrics computed:
    - MMD Degree      (gaussian EMD kernel)
    - MMD Clustering  (gaussian EMD kernel)
    - MMD Orbit       (ORCA 4-node orbit counts)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------

FILENAME_MAP = {
    'graph_rnn_grid':            'training_graphs/GraphRNN_RNN_grid_4_128_train_0.dat',
    'graph_rnn_protein':         'training_graphs/GraphRNN_RNN_protein_4_128_train_0.dat',
    'graph_rnn_ego':             'training_graphs/GraphRNN_RNN_citeseer_4_128_train_0.dat',
    'graph_rnn_community':       'training_graphs/GraphRNN_RNN_caveman_4_128_train_0.dat',
    'graph_rnn_ego_small':       'training_graphs/GraphRNN_RNN_citeseer_small_4_64_train_0.dat',
    'graph_rnn_community_small': 'training_graphs/GraphRNN_RNN_caveman_small_4_64_train_0.dat',
}


def load_test_graphs(dataset):
    filename = FILENAME_MAP[dataset]
    with open(filename, 'rb') as f:
        graphs = pickle.load(f)
    graphs_len = len(graphs)
    test_graphs = graphs[int(0.8 * graphs_len):]
    test_graphs = [g.to_undirected() for g in test_graphs]
    print("Loaded {} test graphs from {}".format(len(test_graphs), filename))
    return test_graphs


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_graphs', required=True,
                        help='Path to graphs.p pickle file of generated graphs')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g. graph_rnn_ego_small)')
    parser.add_argument('--output_file', default='mmd_results.json',
                        help='Where to save MMD results as JSON')
    parser.add_argument('--graphrnn_eval_dir', default='GraphRNN/eval',
                        help='Path to GraphRNN eval directory containing stats.py and orca')
    parser.add_argument('--wandb_project', default='graph-normalising-flow')
    parser.add_argument('--wandb_run_name', default='mmd_eval')
    args = parser.parse_args()

    # Add GraphRNN eval to path
    eval_dir = os.path.abspath(args.graphrnn_eval_dir)
    parent_dir = os.path.dirname(eval_dir)  # GraphRNN/ (needed for "import eval.mmd" and orca)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Set ORCA binary path
    orca_path = os.path.join(eval_dir, 'orca', 'orca')
    os.environ['ORCA_PATH'] = orca_path

    try:
        from stats import degree_stats, clustering_stats, orbit_stats_all
        print("Loaded GraphRNN eval functions from {}".format(eval_dir))
    except ImportError as e:
        print("ERROR: Could not import GraphRNN stats: {}".format(e))
        print("Make sure GraphRNN is cloned at: {}".format(eval_dir))
        raise

    # Load generated graphs
    print("Loading generated graphs from {}...".format(args.generated_graphs))
    with open(args.generated_graphs, 'rb') as f:
        generated_graphs = pickle.load(f)
    print("Loaded {} generated graphs".format(len(generated_graphs)))

    # Load test graphs
    print("Loading test graphs for dataset {}...".format(args.dataset))
    test_graphs = load_test_graphs(args.dataset)

    # Compute MMD Degree
    print("\nComputing MMD Degree...")
    mmd_deg = degree_stats(test_graphs, generated_graphs)
    print("  MMD Degree: {:.6f}".format(mmd_deg))

    # Compute MMD Clustering
    print("Computing MMD Clustering...")
    mmd_clust = clustering_stats(test_graphs, generated_graphs)
    print("  MMD Clustering: {:.6f}".format(mmd_clust))

    # Compute MMD Orbit
    print("Computing MMD Orbit (ORCA)...")
    # Filter graphs with fewer than 4 nodes (ORCA requires at least 4)
    gen_graphs_orbit  = [g for g in generated_graphs if g.number_of_nodes() >= 4]
    test_graphs_orbit = [g for g in test_graphs      if g.number_of_nodes() >= 4]
    print("  Graphs for orbit (generated): {}/{}, (test): {}/{}".format(
          len(gen_graphs_orbit), len(generated_graphs),
          len(test_graphs_orbit), len(test_graphs)))
    # stats.py hardcodes './eval/orca/orca' so must run from GraphRNN/ directory
    orig_dir = os.getcwd()
    os.chdir(parent_dir)
    try:
        mmd_orb = orbit_stats_all(test_graphs_orbit, gen_graphs_orbit)
    finally:
        os.chdir(orig_dir)
    print("  MMD Orbit: {:.6f}".format(mmd_orb))

    results = {
        "mmd_degree":     float(mmd_deg),
        "mmd_clustering": float(mmd_clust),
        "mmd_orbit":      float(mmd_orb),
        "num_generated":  len(generated_graphs),
        "num_test":       len(test_graphs),
        "dataset":        args.dataset,
    }

    print("\n=== MMD Results ===")
    print("MMD Degree:     {:.6f}".format(mmd_deg))
    print("MMD Clustering: {:.6f}".format(mmd_clust))
    print("MMD Orbit:      {:.6f}".format(mmd_orb))
    print("(Lower is better — 0 means perfect match)")

    # Save to JSON
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to {}".format(args.output_file))

    # Log to wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        wandb.log({
            "mmd/degree":     mmd_deg,
            "mmd/clustering": mmd_clust,
            "mmd/orbit":      mmd_orb,
        })
        wandb.finish()
        print("Logged to W&B project: {}".format(args.wandb_project))


if __name__ == '__main__':
    main()
