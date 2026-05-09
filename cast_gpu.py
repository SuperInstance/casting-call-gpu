#!/usr/bin/env python3
"""
Casting-Call GPU — Anchor-Point Signature Mathematics

GPU-accelerated computation of signature distance matrices, voice spline
interpolation, and model clustering for the Casting-Call theory.

Supports CuPy (NVIDIA GPU) and PyTorch (CPU/GPU), with graceful NumPy fallback.

Theory:
    Each piece of text has a signature: a point in N-dimensional anchor space.
    Each model has a set of signatures (its "voice spline").
    The distance between two signatures tells us how similar their texts are.
    The distance matrix across all signatures reveals clusters and outliers.

    When a new task type is proposed, we interpolate along the expected voice
    spline to predict which model would handle it best — before any actual test.

Usage:
    # CLI
    cast-gpu matrix signatures/                # Compute distance matrix
    cast-gpu interpolate task_spec.json         # Spline interpolation
    cast-gpu cluster signatures/ --k 4          # Model clustering

    # Python
    from cast_gpu import SignatureMatrix, VoiceSpline, ModelCluster
    matrix = SignatureMatrix(device='auto')
    dists = matrix.compute(signatures)
"""

import json
import os
import sys
import argparse
import glob
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import numpy as np

# ── Device Detection ────────────────────────────────────────────────────────

def detect_device():
    """Detect best available compute device. Returns (backend, device_name)."""
    # Try CuPy (NVIDIA GPU)
    try:
        import cupy as cp
        if cp.cuda.is_available():
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            return ('cupy', f'GPU ({device_name})')
    except (ImportError, Exception):
        pass

    # Try PyTorch (CPU/GPU)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return ('torch_cuda', f'GPU ({device_name})')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return ('torch_mps', 'Apple MPS')
    except (ImportError, Exception):
        pass

    # Fall back to CPU via NumPy
    return ('numpy', 'CPU (NumPy)')


def compute_backend(device='auto'):
    """Return the appropriate compute library based on device selection."""
    backend, device_name = detect_device()

    if device == 'cpu':
        backend, device_name = ('numpy', 'CPU (NumPy, forced)')

    if backend == 'cupy':
        import cupy as cp
        return cp, device_name

    if backend in ('torch_cuda', 'torch_mps'):
        import torch
        return torch, device_name

    return np, device_name


# ── Signature Loading ───────────────────────────────────────────────────────

def load_signatures(sig_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load all signature JSON files from a directory.

    Returns:
        (signature_vectors, metadata) where signature_vectors is shape (N, M)
        and metadata is a list of dicts with original file info.
    """
    sig_dir = Path(sig_dir)
    if not sig_dir.is_dir():
        raise NotADirectoryError(f"Signature directory not found: {sig_dir}")

    files = sorted(sig_dir.glob('*.json'))
    if not files:
        # Also check subdirs
        files = sorted(sig_dir.rglob('*.json'))

    if not files:
        raise FileNotFoundError(f"No signature JSON files found in {sig_dir}")

    vectors = []
    metadata = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        meta = {
            'file': str(f),
            'text_name': data.get('text_name', f.stem),
            'generating_model': data.get('generating_model', 'unknown'),
        }

        sig_raw = data.get('signature', data)
        ap = data.get('anchor_points', {})

        # Extract numerical features
        if isinstance(sig_raw, dict):
            # New format: dict with numerical features
            vec = features_from_signature(sig_raw)
        elif isinstance(sig_raw, str):
            # Anchor-point format: signature is a compressed string code
            # Extract features from anchor_points dict instead
            if isinstance(ap, dict) and len(ap) > 0:
                vec = features_from_anchor_points(ap)
            else:
                # Decode the compressed string if no anchor points:
                # S-C-A-L-P-M-N-O-N-S → 10 binary features
                parts = sig_raw.split('-')
                try:
                    vec = np.array([float(ord(p[0]) if p else 0.0) for p in parts], dtype=np.float64)
                except (IndexError, ValueError):
                    vec = np.array([float(hash(sig_raw) % 1000) / 1000.0], dtype=np.float64)
        else:
            vec = np.array([0.0], dtype=np.float64)

        # Store anchor points in metadata
        if isinstance(ap, dict):
            for key, val in ap.items():
                if isinstance(val, dict) and 'confidence' in val:
                    meta[f'anchor_{key}'] = val.get('value', str(val))
                    meta[f'anchor_{key}_confidence'] = val.get('confidence', 0.0)
                elif isinstance(val, (int, float)):
                    meta[key] = val

        vectors.append(vec)
        metadata.append(meta)

    if not vectors:
        raise ValueError("No valid signature vectors found")

    # Ensure all vectors have same length (pad shorter ones)
    max_len = max(v.shape[0] for v in vectors)
    padded = []
    for v in vectors:
        if v.shape[0] < max_len:
            pad = np.zeros(max_len - v.shape[0], dtype=np.float64)
            padded.append(np.concatenate([v, pad]))
        else:
            padded.append(v)

    return np.array(padded), metadata


def features_from_anchor_points(ap: dict) -> np.ndarray:
    """
    Extract numerical feature vector from an anchor_points dictionary.

    Each anchor point has: {value, confidence, scale: [options]}
    Encoded as: one-hot at value position (weighted by confidence) + confidence.
    """
    features = []
    for key, val in ap.items():
        if isinstance(val, dict) and 'value' in val:
            confidence = val.get('confidence', 0.5)
            raw = val.get('value', 0)
            scale = val.get('scale', [])

            if scale:
                # One-hot at value position, weighted by confidence
                idx = scale.index(raw) if raw in scale else -1
                for i, option in enumerate(scale):
                    features.append(confidence if i == idx else 0.0)
                features.append(float(confidence))
            elif isinstance(raw, (int, float)):
                # Simple numeric value: encode as value * confidence
                features.append(float(raw) * float(confidence))
                features.append(float(confidence))
            else:
                # String value: hash to a float
                features.append(float(hash(str(raw)) % 1000) / 1000.0 * float(confidence))
                features.append(float(confidence))

    return np.array(features, dtype=np.float64)


def features_from_signature(sig: dict) -> np.ndarray:
    """
    Extract numerical features from a signature dictionary.

    Converts categorical values to numerical via one-hot encoding on the
    known scale dimensions.
    """
    known_categorical = {
        'opening_strategy': ['sensorium', 'abstraction', 'grounded', 'paradox',
                             'theorem', 'narrative', 'quote', 'question', 'declarative'],
        'reader_relationship': ['direct_address', 'impersonal', 'collaborative',
                                'instructive', 'confrontational'],
        'negative_space_use': ['baseline', 'texture', 'insight', 'absent'],
        'tone': ['formal', 'casual', 'technical', 'narrative', 'expository'],
    }

    features = []

    # Add numerical values directly
    for num_key in ['word_count', 'line_count', 'sentence_count',
                    'avg_word_length', 'avg_sentence_length',
                    'lexical_diversity']:
        val = sig.get(num_key, 0)
        features.append(float(val))

    # One-hot encode categorical values
    for cat_key, scale in known_categorical.items():
        val = sig.get(cat_key, scale[0])
        # One-hot
        for option in scale:
            features.append(1.0 if val == option else 0.0)

    return np.array(features)


# ── Signature Matrix Computation ────────────────────────────────────────────

class SignatureMatrix:
    """
    Compute the pairwise distance matrix for a set of signatures.

    Supports:
      - Cosine distance (default — captures direction of style)
      - Euclidean distance (captures magnitude)
      - Custom distance function

    The matrix is symmetric with zeros on the diagonal.
    shape: (N, N) where N = number of signatures
    """

    def __init__(self, device='auto'):
        self.xp, self.device_name = compute_backend(device)
        self._name = 'SignatureMatrix'
        self._backend = type(self.xp).__module__.split('.')[0]

    def compute(self, signatures: np.ndarray, metric='cosine') -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            signatures: ndarray of shape (N, M) where N = signatures, M = features
            metric: 'cosine' or 'euclidean'

        Returns:
            ndarray of shape (N, N) — symmetric distance matrix
        """
        xp = self.xp
        is_torch = hasattr(xp, 'tensor')
        is_cupy = hasattr(xp, 'cupy') or (hasattr(xp, '__name__') and xp.__name__ == 'cupy')

        # Handle numpy, torch, and cupy backends
        if is_torch:
            dev = 'cuda' if xp.cuda.is_available() else 'cpu'
            sigs = xp.tensor(signatures, dtype=xp.float64, device=dev)
        else:
            sigs = xp.array(signatures, dtype=xp.float64)
        n = sigs.shape[0]

        # Matrix multiply function (torch.matmul vs np.dot)
        def matmul(a, b):
            return xp.matmul(a, b) if is_torch else xp.dot(a, b)

        if metric == 'cosine':
            # Normalize to unit vectors
            if is_torch:
                norms = xp.linalg.norm(sigs, dim=1, keepdim=True)
            else:
                norms = xp.linalg.norm(sigs, axis=1, keepdims=True)
            norms = xp.clip(norms, 1e-10, None)  # Avoid div by zero
            unit = sigs / norms

            # Cosine similarity matrix
            sim = matmul(unit, unit.T)
            # Clamp for numerical stability
            sim = xp.clip(sim, -1.0, 1.0)
            # Convert to distance: 1 - cos(angle)
            dist = 1.0 - sim

        elif metric == 'euclidean':
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            norms = xp.sum(sigs ** 2, axis=1, keepdims=True)
            dots = matmul(sigs, sigs.T)
            dist = xp.sqrt(xp.clip(norms + norms.T - 2 * dots, 0, None))

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

        # Zero out diagonal
        if is_torch:
            dist.fill_diagonal_(0.0)
        else:
            xp.fill_diagonal(dist, 0.0)

        # Convert back to numpy for serialization
        if hasattr(dist, 'cpu'):
            return dist.cpu().numpy()
        return dist

    def summarize(self, dist_matrix: np.ndarray, metadata: List[Dict]) -> Dict:
        """
        Analyze the distance matrix and return summary statistics.

        For each pair, find minimum, maximum, mean, and interesting outliers.
        """
        n = dist_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        distances = dist_matrix[mask]

        # Most similar pair
        flat = dist_matrix.copy()
        np.fill_diagonal(flat, np.inf)
        min_idx = np.unravel_index(np.argmin(flat), flat.shape)
        most_similar = {
            'a': metadata[min_idx[0]].get('text_name', f'sig_{min_idx[0]}'),
            'b': metadata[min_idx[1]].get('text_name', f'sig_{min_idx[1]}'),
            'distance': float(flat[min_idx]),
        }

        # Least similar pair (use a copy without the diagonal filled to Inf)
        flat_max = dist_matrix.copy()
        np.fill_diagonal(flat_max, -1.0)
        max_idx = np.unravel_index(np.argmax(flat_max), flat_max.shape)
        least_similar = {
            'a': metadata[max_idx[0]].get('text_name', f'sig_{max_idx[0]}'),
            'b': metadata[max_idx[1]].get('text_name', f'sig_{max_idx[1]}'),
            'distance': float(flat_max[max_idx]),
        }

        # Per-signature stats
        signature_stats = []
        for i in range(n):
            others = np.concatenate([dist_matrix[i, :i], dist_matrix[i, i+1:]])
            signature_stats.append({
                'name': metadata[i].get('text_name', f'sig_{i}'),
                'min_distance': float(others.min()),
                'max_distance': float(others.max()),
                'mean_distance': float(others.mean()),
                'std_distance': float(others.std()),
            })

        return {
            'shape': (n, n),
            'metric': 'cosine',
            'total_distances': len(distances),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'most_similar': most_similar,
            'least_similar': least_similar,
            'per_signature': signature_stats,
        }


# ── Voice Spline Interpolation ──────────────────────────────────────────────

class VoiceSpline:
    """
    Interpolate the expected signature for a new task type given N anchor points
    from a model's known outputs.

    Given:
        - N task types with known signatures for a model
        - A new task type position in the task-space

    Produces:
        - Interpolated signature for the new task type
        - Confidence estimate based on proximity to known anchors
    """

    def __init__(self, device='auto'):
        from scipy import interpolate
        self.interpolate = interpolate
        self.xp, self.device_name = compute_backend(device)

    def fit_predict(self,
                    task_positions: np.ndarray,
                    known_signatures: np.ndarray,
                    query_position: np.ndarray) -> Dict:
        """
        Interpolate a signature at a new task position.

        Args:
            task_positions: (N, D) array of known task coordinates
            known_signatures: (N, M) array of known signature vectors
            query_position: (D,) array of the new task's position

        Returns:
            dict with interpolated signature and confidence
        """
        m = known_signatures.shape[1]
        interpolated = np.zeros(m)

        # Interpolate each feature dimension independently
        for dim in range(m):
            try:
                f = self.interpolate.RBFInterpolator(
                    task_positions, known_signatures[:, dim],
                    kernel='linear'
                )
                interpolated[dim] = float(f(query_position.reshape(1, -1)))
            except Exception:
                # Fall back to nearest-neighbor
                distances = np.linalg.norm(task_positions - query_position, axis=1)
                nearest = np.argmin(distances)
                interpolated[dim] = known_signatures[nearest, dim]

        # Confidence: based on proximity to nearest anchor point
        distances = np.linalg.norm(task_positions - query_position, axis=1)
        min_dist = distances.min()
        # Confidence decays with distance: 1.0 at 0 distance, 0.0 at distance > 2.0
        confidence = float(np.clip(1.0 - min_dist / 2.0, 0.0, 1.0))

        return {
            'interpolated_signature': interpolated.tolist(),
            'confidence': confidence,
            'nearest_anchor_distance': float(min_dist),
            'n_anchors': task_positions.shape[0],
        }


# ── Model Clustering ────────────────────────────────────────────────────────

class ModelCluster:
    """
    Given distance matrices from multiple models, cluster models by
    signature similarity. Models that produce similar signatures for
    the same inputs cluster together.

    Also detects "signature drift" — models whose signatures shift
    significantly across similar task types.
    """

    def __init__(self, device='auto'):
        self.xp, self.device_name = compute_backend(device)

    def cluster_by_model(self,
                         model_name: str,
                         all_signatures: Dict[str, np.ndarray],
                         n_clusters: int = 3) -> Dict:
        """
        Cluster the signatures produced by a single model across different tasks.

        Args:
            model_name: The model identifier
            all_signatures: dict mapping model_name -> (N, M) signature array
            n_clusters: number of clusters (best guess)

        Returns:
            dict with cluster assignments and centroid signatures
        """
        sigs = all_signatures.get(model_name)
        if sigs is None:
            raise KeyError(f"Model '{model_name}' not found in signature data")

        # Compute distance matrix
        matrix = SignatureMatrix(device=self.xp.__name__)
        dist_mat = matrix.compute(sigs)

        # Simple k-means clustering (avoid sklearn dependency)
        n, m = sigs.shape
        k = min(n_clusters, n)

        # Initialize with k-means++
        rng = np.random.RandomState(42)
        centroids = [sigs[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.linalg.norm(s - c) for c in centroids) for s in sigs])
            probs = dists / dists.sum()
            centroids.append(sigs[rng.choice(n, p=probs)])
        centroids = np.array(centroids)

        # Iterate until convergence
        labels = np.zeros(n, dtype=int)
        for _ in range(100):
            prev_labels = labels.copy()
            # Assign
            for i in range(n):
                labels[i] = int(np.argmin([np.linalg.norm(sigs[i] - c) for c in centroids]))
            # Update
            for j in range(k):
                members = sigs[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
            if np.all(labels == prev_labels):
                break

        # Build result
        clusters = {}
        for j in range(k):
            members = np.where(labels == j)[0]
            clusters[f'cluster_{j}'] = {
                'size': int(len(members)),
                'indices': members.tolist(),
                'centroid_signature': centroids[j].tolist(),
            }

        return {
            'model': model_name,
            'n_clusters': k,
            'n_signatures': n,
            'clusters': clusters,
            'labels': labels.tolist(),
        }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Casting-Call GPU — Signature Mathematics Engine'
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # matrix
    p_matrix = sub.add_parser('matrix', help='Compute signature distance matrix')
    p_matrix.add_argument('signature_dir', type=str, help='Directory of signature JSON files')
    p_matrix.add_argument('--metric', choices=['cosine', 'euclidean'], default='cosine')
    p_matrix.add_argument('--output', '-o', type=str, help='Output JSON path')
    p_matrix.add_argument('--device', choices=['auto', 'cpu'], default='auto')

    # interpolate
    p_interp = sub.add_parser('interpolate', help='Spline interpolation for task signatures')
    p_interp.add_argument('signature_dir', type=str, help='Directory of signature JSON files')
    p_interp.add_argument('--device', choices=['auto', 'cpu'], default='auto')

    # cluster
    p_cluster = sub.add_parser('cluster', help='Cluster signatures by model')
    p_cluster.add_argument('signature_dir', type=str, help='Directory of signature JSON files')
    p_cluster.add_argument('--k', type=int, default=3, help='Number of clusters')
    p_cluster.add_argument('--device', choices=['auto', 'cpu'], default='auto')

    # info
    p_info = sub.add_parser('info', help='Show device and system info')

    args = parser.parse_args()

    if args.command == 'info':
        backend, device_name = detect_device()
        print(f"Detected backend: {backend}")
        print(f"Device: {device_name}")
        return

    # Load signatures
    try:
        sigs, meta = load_signatures(args.signature_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error loading signatures: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(sigs)} signatures with {sigs.shape[1]} features each", file=sys.stderr)
    backend_name = detect_device()[1]
    print(f"Using: {backend_name}", file=sys.stderr)

    if args.command == 'matrix':
        matrix = SignatureMatrix(device=args.device)
        dist = matrix.compute(sigs, metric=args.metric)
        summary = matrix.summarize(dist, meta)
        summary['device'] = backend_name

        output = args.output
        if not output:
            output = 'signature_matrix.json'

        with open(output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Matrix saved to {output}")
        print(json.dumps(summary, indent=2))

    elif args.command == 'cluster':
        # Group signatures by model
        from collections import defaultdict
        grouped = defaultdict(list)
        for i, m in enumerate(meta):
            model_name = m.get('generating_model', 'unknown')
            grouped[model_name].append(i)

        model_sigs = {}
        for model_name, indices in grouped.items():
            model_sigs[model_name] = sigs[indices]

        clusterer = ModelCluster(device=args.device)
        results = {}
        for model_name in model_sigs:
            result = clusterer.cluster_by_model(
                model_name, model_sigs, n_clusters=min(args.k, len(model_sigs[model_name]))
            )
            results[model_name] = result

        output = 'cluster_results.json'
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Cluster results saved to {output}")
        print(json.dumps(results, indent=2))

    elif args.command == 'interpolate':
        # Simple interpolation: use all signatures as anchors for random queries
        print("Voice spline interpolation requires task positions.")
        print("Use the python API for custom interpolation.", file=sys.stderr)
        print(f"Available: {[m.get('text_name') for m in meta]}", file=sys.stderr)


if __name__ == '__main__':
    main()
