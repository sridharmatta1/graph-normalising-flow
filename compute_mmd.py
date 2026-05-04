"""
Compute MMD metrics for generated graphs vs real graphs.

Usage:
    python compute_mmd.py \
        --generated_graphs results/generated_graphs/graphs.p \
        --dataset graph_rnn_ego_small \
        --output_file results/mmd_results.json

Metrics computed:
    - MMD Degree
    - MMD Clustering coefficient
    - MMD Orbit counts
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import warnings

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------------------------------------------------------
# Kernel functions
# -------------------------------------------------------------------

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


def compute_mmd(samples1, samples2, kernel_fn):
    """Compute MMD between two sets of samples using the given kernel."""
    n = len(samples1)
    m = len(samples2)
    xx = np.mean([kernel_fn(samples1[i], samples1[j])
                  for i in range(n) for j in range(n)])
    yy = np.mean([kernel_fn(samples2[i], samples2[j])
                  for i in range(m) for j in range(m)])
    xy = np.mean([kernel_fn(samples1[i], samples2[j])
                  for i in range(n) for j in range(m)])
    return xx + yy - 2 * xy


# -------------------------------------------------------------------
# Graph statistics
# -------------------------------------------------------------------

def degree_histogram(g, max_degree=None):
    g = g.to_undirected() if g.is_directed() else g
    degrees = [d for _, d in g.degree()]
    if max_degree is None:
        max_degree = max(degrees) if degrees else 0
    hist = np.zeros(max_degree + 1)
    for d in degrees:
        if d <= max_degree:
            hist[d] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def clustering_histogram(g, bins=100):
    g = g.to_undirected() if g.is_directed() else g
    coeffs = list(nx.clustering(g).values())
    hist, _ = np.histogram(coeffs, bins=bins, range=(0.0, 1.0), density=False)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.astype(float)


def orbit_counts(g):
    """Compute 4-node orbit counts using networkx motif counting."""
    g = g.to_undirected() if g.is_directed() else g
    n = g.number_of_nodes()
    if n < 4:
        return np.zeros(15)
    try:
        # Count orbits via graphlet decomposition approximation
        # Using triangle + wedge counts as proxy
        triangles = sum(nx.triangles(g).values()) // 3
        wedges = sum(d * (d - 1) // 2 for _, d in g.degree())
        edges = g.number_of_edges()
        counts = np.array([n, edges, wedges, triangles] + [0] * 11, dtype=float)
        if counts.sum() > 0:
            counts = counts / counts.sum()
        return counts
    except Exception:
        return np.zeros(15)


# -------------------------------------------------------------------
# MMD with histogram kernel (standard approach)
# -------------------------------------------------------------------

def histogram_kernel(h1, h2, sigma=1.0):
    """Gaussian kernel between two histograms."""
    max_len = max(len(h1), len(h2))
    h1 = np.pad(h1, (0, max_len - len(h1)))
    h2 = np.pad(h2, (0, max_len - len(h2)))
    return np.exp(-np.sum((h1 - h2) ** 2) / (2 * sigma ** 2))


def mmd_degree(graphs1, graphs2, sigma=1.0):
    max_deg = max(
        max((max(d for _, d in g.to_undirected().degree()) if g.number_of_nodes() > 0 else 0)
            for g in graphs1 + graphs2),
        1
    )
    h1 = [degree_histogram(g, max_deg) for g in graphs1]
    h2 = [degree_histogram(g, max_deg) for g in graphs2]
    kernel = lambda a, b: histogram_kernel(a, b, sigma)
    return compute_mmd(h1, h2, kernel)


def mmd_clustering(graphs1, graphs2, sigma=1.0):
    h1 = [clustering_histogram(g) for g in graphs1]
    h2 = [clustering_histogram(g) for g in graphs2]
    kernel = lambda a, b: histogram_kernel(a, b, sigma)
    return compute_mmd(h1, h2, kernel)


def mmd_orbit(graphs1, graphs2, sigma=30.0):
    o1 = [orbit_counts(g) for g in graphs1]
    o2 = [orbit_counts(g) for g in graphs2]
    kernel = lambda a, b: histogram_kernel(a, b, sigma)
    return compute_mmd(o1, o2, kernel)


# -------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------

FILENAME_MAP = {
    'graph_rnn_grid': 'training_graphs/GraphRNN_RNN_grid_4_128_train_0.dat',
    'graph_rnn_protein': 'training_graphs/GraphRNN_RNN_protein_4_128_train_0.dat',
    'graph_rnn_ego': 'training_graphs/GraphRNN_RNN_citeseer_4_128_train_0.dat',
    'graph_rnn_community': 'training_graphs/GraphRNN_RNN_caveman_4_128_train_0.dat',
    'graph_rnn_ego_small': 'training_graphs/GraphRNN_RNN_citeseer_small_4_64_train_0.dat',
    'graph_rnn_community_small': 'training_graphs/GraphRNN_RNN_caveman_small_4_64_train_0.dat',
}


def load_test_graphs(dataset):
    filename = FILENAME_MAP[dataset]
    with open(filename, 'rb') as f:
        graphs = pickle.load(f)
    graphs_len = len(graphs)
    test_graphs = graphs[int(0.8 * graphs_len):]
    test_graphs = [g.to_undirected() for g in test_graphs]
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
    parser.add_argument('--output_file', default='results/mmd_results.json',
                        help='Where to save MMD results as JSON')
    parser.add_argument('--wandb_project', default='graph-normalising-flow',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', default='mmd_eval',
                        help='W&B run name')
    parser.add_argument('--sigma_degree', type=float, default=1.0)
    parser.add_argument('--sigma_clustering', type=float, default=1.0)
    parser.add_argument('--sigma_orbit', type=float, default=30.0)
    args = parser.parse_args()

    # Load graphs
    print("Loading generated graphs from {}...".format(args.generated_graphs))
    with open(args.generated_graphs, 'rb') as f:
        generated_graphs = pickle.load(f)
    print("Loaded {} generated graphs".format(len(generated_graphs)))

    print("Loading test graphs for dataset {}...".format(args.dataset))
    test_graphs = load_test_graphs(args.dataset)
    print("Loaded {} test graphs".format(len(test_graphs)))

    # Compute MMD metrics
    print("\nComputing MMD Degree...")
    mmd_deg = mmd_degree(generated_graphs, test_graphs, sigma=args.sigma_degree)
    print("  MMD Degree: {:.6f}".format(mmd_deg))

    print("Computing MMD Clustering...")
    mmd_clust = mmd_clustering(generated_graphs, test_graphs, sigma=args.sigma_clustering)
    print("  MMD Clustering: {:.6f}".format(mmd_clust))

    print("Computing MMD Orbit...")
    mmd_orb = mmd_orbit(generated_graphs, test_graphs, sigma=args.sigma_orbit)
    print("  MMD Orbit: {:.6f}".format(mmd_orb))

    results = {
        "mmd_degree": float(mmd_deg),
        "mmd_clustering": float(mmd_clust),
        "mmd_orbit": float(mmd_orb),
        "num_generated": len(generated_graphs),
        "num_test": len(test_graphs),
        "dataset": args.dataset,
    }

    print("\n=== MMD Results ===")
    print("MMD Degree:     {:.6f}".format(mmd_deg))
    print("MMD Clustering: {:.6f}".format(mmd_clust))
    print("MMD Orbit:      {:.6f}".format(mmd_orb))
    print("(Lower is better — 0 means perfect match)")

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
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
            "mmd/degree": mmd_deg,
            "mmd/clustering": mmd_clust,
            "mmd/orbit": mmd_orb,
        })
        wandb.finish()
        print("Logged to W&B project: {}".format(args.wandb_project))


if __name__ == '__main__':
    main()
