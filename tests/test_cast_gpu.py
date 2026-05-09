"""Tests for the Casting-Call GPU engine."""

import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from cast_gpu import (
    SignatureMatrix,
    VoiceSpline,
    ModelCluster,
    load_signatures,
    features_from_signature,
    compute_backend,
    detect_device,
)


def test_features_from_signature():
    """Test that feature extraction produces a consistent-length vector."""
    sig = {
        'word_count': 100,
        'line_count': 10,
        'sentence_count': 8,
        'avg_word_length': 4.5,
        'avg_sentence_length': 12.3,
        'lexical_diversity': 0.65,
        'opening_strategy': 'narrative',
        'reader_relationship': 'collaborative',
        'negative_space_use': 'texture',
        'tone': 'narrative',
    }
    vec = features_from_signature(sig)
    assert vec.ndim == 1, "Feature vector should be 1D"
    assert vec.dtype == np.float64, "Should be float64"
    assert len(vec) > 0, "Should have features"


def test_signature_matrix_cosine():
    """Test cosine distance matrix computation."""
    # 3 signatures with 5 features
    sigs = np.array([
        [1.0, 0.0, 0.5, 0.2, 0.1],
        [0.9, 0.1, 0.4, 0.3, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.8],
    ])

    mat = SignatureMatrix(device='cpu')
    dist = mat.compute(sigs, metric='cosine')

    assert dist.shape == (3, 3), f"Expected (3,3), got {dist.shape}"
    assert np.allclose(np.diag(dist), 0), "Diagonal should be zero"
    assert np.all(dist >= 0), "Distances should be non-negative"
    assert np.all(dist <= 2.0), "Cosine distance should be in [0, 2]"


def test_signature_matrix_euclidean():
    """Test euclidean distance matrix."""
    sigs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    mat = SignatureMatrix(device='cpu')
    dist = mat.compute(sigs, metric='euclidean')

    assert dist.shape == (2, 2)
    assert np.allclose(np.diag(dist), 0)
    assert np.isclose(dist[0, 1], np.sqrt(2)), "Euclidean( [1,0], [0,1] ) = sqrt(2)"


def test_signature_matrix_symmetry():
    """Distance matrix should be symmetric."""
    rng = np.random.RandomState(42)
    sigs = rng.rand(10, 8)

    mat = SignatureMatrix(device='cpu')
    dist = mat.compute(sigs, metric='cosine')

    assert np.allclose(dist, dist.T), "Distance matrix must be symmetric"


def test_load_signatures():
    """Test loading signatures from a directory."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in ['sig_a.json', 'sig_b.json']:
            with open(Path(tmp) / name, 'w') as f:
                json.dump({
                    'text_name': name.replace('.json', ''),
                    'signature': {
                        'word_count': 100,
                        'line_count': 10,
                        'sentence_count': 8,
                        'avg_word_length': 4.5,
                        'avg_sentence_length': 12.3,
                        'lexical_diversity': 0.65,
                        'opening_strategy': 'narrative',
                        'reader_relationship': 'collaborative',
                        'negative_space_use': 'texture',
                    },
                }, f)

        sigs, meta = load_signatures(tmp)
        assert sigs.shape[0] == 2, f"Expected 2 signatures, got {sigs.shape[0]}"
        assert len(meta) == 2, f"Expected 2 metadata entries, got {len(meta)}"


def test_summarize():
    """Test matrix summary generation."""
    sigs = np.eye(3)  # 3 orthogonal signatures
    mat = SignatureMatrix(device='cpu')
    dist = mat.compute(sigs, metric='cosine')

    meta = [{'text_name': 'a'}, {'text_name': 'b'}, {'text_name': 'c'}]
    summary = mat.summarize(dist, meta)

    assert 'min_distance' in summary
    assert 'max_distance' in summary
    assert 'mean_distance' in summary
    assert 'most_similar' in summary
    assert 'least_similar' in summary
    assert 'per_signature' in summary
    assert len(summary['per_signature']) == 3


def test_voice_spline():
    """Test voice spline interpolation."""
    # 4 known task positions in 2D space
    task_pos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # Each generates a 3D signature
    known_sigs = np.array([
        [1.0, 0.5, 0.2],
        [0.8, 0.6, 0.3],
        [0.7, 0.4, 0.5],
        [0.9, 0.5, 0.4],
    ])

    spline = VoiceSpline(device='cpu')
    result = spline.fit_predict(task_pos, known_sigs, np.array([0.5, 0.5]))

    assert 'interpolated_signature' in result
    assert 'confidence' in result
    assert len(result['interpolated_signature']) == 3
    assert 0 <= result['confidence'] <= 1


def test_model_cluster():
    """Test model clustering."""
    # Create 2 clear clusters
    cluster_a = np.random.RandomState(1).randn(5, 4) + np.array([2, 2, 2, 2])
    cluster_b = np.random.RandomState(2).randn(5, 4) + np.array([-2, -2, -2, -2])
    sigs = np.vstack([cluster_a, cluster_b])

    model_sigs = {'test_model': sigs}

    clusterer = ModelCluster(device='cpu')
    result = clusterer.cluster_by_model('test_model', model_sigs, n_clusters=2)

    assert result['model'] == 'test_model'
    assert result['n_clusters'] == 2
    assert 'clusters' in result
    assert 'cluster_0' in result['clusters']
    assert 'cluster_1' in result['clusters']


def test_detect_device():
    """Device detection should always return something."""
    backend, device_name = detect_device()
    assert backend in ('numpy', 'cupy', 'torch_cuda', 'torch_mps')
    assert isinstance(device_name, str)
    assert len(device_name) > 0


def test_compute_backend():
    """Compute backend should always return a usable library."""
    xp, name = compute_backend('auto')
    assert hasattr(xp, 'array'), "Backend should be array-capable"

    xp_cpu, _ = compute_backend('cpu')
    assert hasattr(xp_cpu, 'array'), "CPU backend should be array-capable"


if __name__ == '__main__':
    test_features_from_signature()
    test_signature_matrix_cosine()
    test_signature_matrix_euclidean()
    test_signature_matrix_symmetry()
    test_load_signatures()
    test_summarize()
    test_voice_spline()
    test_model_cluster()
    test_detect_device()
    test_compute_backend()
    print("All tests passed! ✅")
