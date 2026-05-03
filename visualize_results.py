"""
Visualize and analyze generated graphs from the GRevNet pipeline.

Usage:
    python visualize_results.py --graphs_file ~/Downloads/generated_graphs/graphs.p \
                                 --output_dir ~/Downloads/generated_graphs/
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np


def graph_stats(g):
    g_undirected = g.to_undirected() if g.is_directed() else g
    n = g_undirected.number_of_nodes()
    e = g_undirected.number_of_edges()
    density = nx.density(g_undirected)
    components = nx.number_connected_components(g_undirected)
    avg_degree = np.mean([d for _, d in g_undirected.degree()]) if n > 0 else 0
    clustering = nx.average_clustering(g_undirected) if n > 0 else 0
    return {
        'nodes': n,
        'edges': e,
        'density': density,
        'components': components,
        'avg_degree': avg_degree,
        'clustering': clustering,
    }


def draw_graph(g, ax, title=None):
    g_undirected = g.to_undirected() if g.is_directed() else g
    try:
        pos = nx.spring_layout(g_undirected, k=0.8, iterations=100, seed=42)
    except Exception:
        pos = nx.random_layout(g_undirected)
    nx.draw(g_undirected, pos, ax=ax,
            node_color='#7b68ee', node_size=80,
            edge_color='#555555', width=1.5,
            with_labels=False, alpha=0.85)
    if title:
        ax.set_title(title, fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs_file', required=True,
                        help='Path to graphs.p pickle file')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save output figures')
    parser.add_argument('--num_show', type=int, default=16,
                        help='Number of graphs to show in the grid')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.graphs_file, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} generated graphs")

    # --- Compute stats ---
    all_stats = [graph_stats(g) for g in graphs]
    keys = ['nodes', 'edges', 'density', 'components', 'avg_degree', 'clustering']
    print("\n=== Graph Statistics ===")
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    for k in keys:
        vals = [s[k] for s in all_stats]
        print(f"{k:<20} {np.mean(vals):>10.3f} {np.std(vals):>10.3f} "
              f"{np.min(vals):>10.3f} {np.max(vals):>10.3f}")

    # --- Grid of graph visualizations ---
    n_show = min(args.num_show, len(graphs))
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n_show:
            s = all_stats[i]
            title = f"G{i}  n={s['nodes']} e={s['edges']}"
            draw_graph(graphs[i], ax, title)
        else:
            ax.axis('off')
    fig.suptitle(f"Generated Graphs (showing {n_show} of {len(graphs)})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, 'generated_graphs_grid.png')
    plt.savefig(grid_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"\nSaved grid figure → {grid_path}")

    # --- Degree distribution histogram ---
    all_degrees = []
    for g in graphs:
        g_u = g.to_undirected() if g.is_directed() else g
        all_degrees.extend([d for _, d in g_u.degree()])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_degrees, bins=range(0, max(all_degrees) + 2),
            color='#7b68ee', edgecolor='white', align='left')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Degree Distribution of Generated Graphs', fontsize=13)
    ax.grid(axis='y', alpha=0.4)
    deg_path = os.path.join(args.output_dir, 'degree_distribution.png')
    plt.tight_layout()
    plt.savefig(deg_path, dpi=120)
    plt.close()
    print(f"Saved degree distribution → {deg_path}")

    # --- Node count distribution ---
    node_counts = [s['nodes'] for s in all_stats]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(node_counts, bins=20, color='#e07b54', edgecolor='white')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Node Count Distribution of Generated Graphs', fontsize=13)
    ax.grid(axis='y', alpha=0.4)
    nodes_path = os.path.join(args.output_dir, 'node_count_distribution.png')
    plt.tight_layout()
    plt.savefig(nodes_path, dpi=120)
    plt.close()
    print(f"Saved node count distribution → {nodes_path}")


if __name__ == '__main__':
    main()
