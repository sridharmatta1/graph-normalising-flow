import json
import os
import numpy as np

results_dir = "/home/matta/graph-normalising-flow/results/ego"
all_degree = []
all_clustering = []
all_orbit = []

for seed in range(1, 6):
    path = f"{results_dir}/seed_{seed}/mmd_results.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        all_degree.append(data["mmd_degree"])
        all_clustering.append(data["mmd_clustering"])
        all_orbit.append(data["mmd_orbit"])
        print(f"Seed {seed}: Degree={data['mmd_degree']:.6f}, "
              f"Clustering={data['mmd_clustering']:.6f}, "
              f"Orbit={data['mmd_orbit']:.6e}")
    else:
        print(f"MISSING: seed_{seed}/mmd_results.json")

print("\n========== FINAL AVERAGE ego-small (5 seeds) ==========")
if all_degree:
    print(f"MMD Degree:     {np.mean(all_degree):.6f} +/- {np.std(all_degree):.6f}")
    print(f"MMD Clustering: {np.mean(all_clustering):.6f} +/- {np.std(all_clustering):.6f}")
    print(f"MMD Orbit:      {np.mean(all_orbit):.6e} +/- {np.std(all_orbit):.6e}")
    print(f"Total runs: {len(all_degree)}")
else:
    print("No results found — check paths above.")

