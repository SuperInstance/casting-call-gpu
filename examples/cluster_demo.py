#!/usr/bin/env python3
"""
Demo: Load signatures from the casting-call repo, compute distance matrix,
and cluster by model.

Usage:
    python examples/cluster_demo.py /tmp/casting-call/signatures/
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from cast_gpu import SignatureMatrix, ModelCluster, load_signatures


def main():
    sig_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not sig_dir:
        print("Usage: python examples/cluster_demo.py <signature_dir>")
        sys.exit(1)

    print(f"Loading signatures from {sig_dir}...")
    sigs, meta = load_signatures(sig_dir)
    print(f"Loaded {len(sigs)} signatures with {sigs.shape[1]} features each")

    # Step 1: Distance matrix
    print("\n=== Step 1: Distance Matrix ===")
    mat = SignatureMatrix(device='auto')
    dist = mat.compute(sigs, metric='cosine')
    summary = mat.summarize(dist, meta)

    print(f"Mean distance: {summary['mean_distance']:.3f}")
    print(f"Most similar pair: {summary['most_similar']}")
    print(f"Least similar pair: {summary['least_similar']}")

    with open('demo_matrix.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Matrix saved to demo_matrix.json")

    # Step 2: Cluster by model
    print("\n=== Step 2: Model Clustering ===")
    grouped = defaultdict(list)
    for i, m in enumerate(meta):
        model_name = m.get('generating_model', 'unknown')
        grouped[model_name].append(i)

    model_sigs = {}
    for model_name, indices in grouped.items():
        model_sigs[model_name] = sigs[indices]
        print(f"  {model_name}: {len(indices)} signatures")

    clusterer = ModelCluster(device='auto')
    all_clusters = {}
    for model_name in model_sigs:
        print(f"\nClustering {model_name}...")
        n = len(model_sigs[model_name])
        k = min(3, n)
        result = clusterer.cluster_by_model(model_name, model_sigs, n_clusters=k)
        all_clusters[model_name] = result['clusters']
        for cid, cdata in result['clusters'].items():
            print(f"  {cid}: {cdata['size']} signatures")

    with open('demo_clusters.json', 'w') as f:
        json.dump(all_clusters, f, indent=2)
    print("Clusters saved to demo_clusters.json")

    print("\nDone! Output files: demo_matrix.json, demo_clusters.json")


if __name__ == '__main__':
    main()
