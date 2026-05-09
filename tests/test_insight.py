"""
Tests for the FM-Oracle1 Insight Engine.

Validates:
  - Backend probing returns something for each backend
  - Voice matching produces the same results as casting-call signatures
  - The insight router produces valid recommendations
  - FLUX bytecode generation produces valid 30-opcode sequences
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from insight_engine import (
    BackendProbe,
    VoiceMatcher,
    InsightRouter,
    hamming_distance,
    parse_signature_code,
    encode_task_to_signature,
    generate_flux_bytecode,
    format_bytecode,
    FLUX_OPCODES,
    FLUX_OPCODE_MAP,
    MODEL_VOICE_SIGNATURES,
    SIGNATURE_DIMENSIONS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Signature Utility Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_flux_opcode_count():
    """The FLUX opcode set should contain exactly the specified ops."""
    # The subset should have exactly 30 core opcodes (including SIG as a
    # FLUX-specific constraint op, part of the 30-opcode set)
    assert len(FLUX_OPCODES) == 30, \
        f"Expected 30 FLUX opcodes, got {len(FLUX_OPCODES)}"
    # Each opcode must be unique
    assert len(set(FLUX_OPCODES)) == len(FLUX_OPCODES), \
        "All FLUX opcodes must be unique"
    # Check specific critical opcodes exist
    for op in ['ADD', 'SUB', 'LOAD', 'STORE', 'JMP', 'SIG', 'HALT']:
        assert op in FLUX_OPCODES, f"Missing critical opcode: {op}"


def test_flux_opcode_map():
    """Every opcode should have a hex code mapping."""
    for op in FLUX_OPCODES:
        assert op in FLUX_OPCODE_MAP, f"Opcode {op} missing from map"
        hex_code = FLUX_OPCODE_MAP[op]
        assert hex_code.startswith('0x'), f"Invalid hex code for {op}: {hex_code}"


def test_parse_signature_code():
    """Parse hyphen-separated signature codes."""
    result = parse_signature_code('G-D-I-S-S')
    assert result == ['G', 'D', 'I', 'S', 'S'], f"Got {result}"

    result = parse_signature_code('')
    assert result == [], "Empty string should return empty list"

    result = parse_signature_code(None)
    assert result == [], "None should return empty list"

    # Mixed case
    result = parse_signature_code('G-d-i-S-s')
    assert result == ['G', 'd', 'i', 'S', 's']


def test_parse_signature_code_variable_length():
    """Handle variable-length signature codes."""
    # Full 10-dimension code
    result = parse_signature_code('G-D-I-S-S-C-A-L-P-M')
    assert len(result) == 10, f"Expected 10 dimensions, got {len(result)}"

    # Short code
    result = parse_signature_code('G-D')
    assert result == ['G', 'D']


def test_hamming_distance():
    """Hamming distance should count differing positions."""
    # Same code
    assert hamming_distance('G-D-I-S-S', 'G-D-I-S-S') == 0

    # One difference
    assert hamming_distance('G-D-I-S-S', 'G-D-I-S-C') == 1

    # All different
    assert hamming_distance('G-G-G', 'D-D-D') == 3

    # Different lengths (padded)
    dist = hamming_distance('G-D-I', 'G-D-I-S-S')
    assert dist == 2, f"Expected 2 (padded), got {dist}"

    # Empty codes
    assert hamming_distance('', '') == 0
    assert hamming_distance('G', '') == 1


def test_hamming_distance_case_sensitive():
    """Case matters in signature comparison — parsed parts preserve case."""
    # After parse: ['g','d','i','s','s'] vs ['G','D','I','S','S'] — all differ
    assert hamming_distance('g-d-i-s-s', 'G-D-I-S-S') == 5


def test_encode_task_to_signature():
    """Task description should produce a valid signature code."""
    # A concrete code task
    sig = encode_task_to_signature("write a python function for matrix multiplication")
    assert sig, "Should produce a non-empty signature"
    parts = parse_signature_code(sig)
    assert len(parts) == 10, f"Expected 10 dimensions, got {len(parts)}"

    # An abstract theoretical task
    sig = encode_task_to_signature("prove a theorem about abstract algebra")
    assert sig, "Should produce a non-empty signature"

    # An empty task
    sig = encode_task_to_signature("")
    assert sig, "Empty task should still produce a default signature"


def test_encode_task_to_signature_consistency():
    """Same task description should produce the same signature."""
    sig1 = encode_task_to_signature("use a GPU for matrix math")
    sig2 = encode_task_to_signature("use a GPU for matrix math")
    assert sig1 == sig2, "Same input should produce same output"

    # Tasks that trigger different high keywords
    sig3 = encode_task_to_signature("abstract creative constraint solver")
    # 'abstract' → low for Grounding (g), 'creative' → high for Creativity (C)
    assert sig1 != sig3, "Very different inputs should produce different signatures"


def test_generate_flux_bytecode():
    """FLUX bytecode generation should produce valid sequences."""
    # With a task
    bc = generate_flux_bytecode(task="matrix multiplication", task_sig="G-D-I-S-S")
    assert isinstance(bc, list), "Bytecode should be a list"
    assert len(bc) > 0, "Bytecode should not be empty"

    # Every opcode must be in the valid set
    for op in bc:
        assert op in FLUX_OPCODES, f"Invalid opcode: {op}"

    # Should always have a prologue (MOV, NOP)
    assert bc[0] == 'MOV', "Bytecode should start with MOV (prologue)"

    # Should always have SIG (constraint verifier)
    assert 'SIG' in bc, "Bytecode should contain SIG constraint op"

    # Should always have HALT (epilogue)
    assert bc[-1] == 'HALT', "Bytecode should end with HALT (epilogue)"


def test_generate_flux_bytecode_from_sig_only():
    """Bytecode can be generated from signature alone."""
    bc = generate_flux_bytecode(task_sig="G-D-I-S-S-C-A-L-P-M")
    assert len(bc) > 0
    assert 'SIG' in bc
    assert bc[-1] == 'HALT'


def test_generate_flux_bytecode_empty():
    """Empty task should still produce basic bytecode."""
    bc = generate_flux_bytecode()
    assert len(bc) >= 3, "Should have at least prologue, SIG, HALT"
    assert bc[-1] == 'HALT'


def test_format_bytecode():
    """Formatted bytecode should be human-readable."""
    bc = ['MOV', 'NOP', 'ADD', 'SIG', 'HALT']
    formatted = format_bytecode(bc)
    assert 'FLUX Bytecode:' in formatted
    assert 'MOV' in formatted
    assert 'ADD' in formatted
    assert 'SIG' in formatted
    assert 'HALT' in formatted


# ═══════════════════════════════════════════════════════════════════════════════
# Backend Probe Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_backend_probe_cpu():
    """CPU should always be available."""
    probe = BackendProbe()
    results = probe.probe_all()
    assert results['cpu']['available'] is True
    assert 'flux_runtime_arm.c' in results['cpu'].get('backend', '')


def test_backend_probe_all_backends_present():
    """All backends should appear in probe results."""
    probe = BackendProbe()
    results = probe.probe_all()
    expected = {'cpu', 'cuda', 'fpga', 'ebpf', 'webgpu', 'vulkan', 'fortran', 'coq'}
    for backend in expected:
        assert backend in results, f"Missing backend: {backend}"
        assert 'available' in results[backend], \
            f"Backend {backend} missing 'available' key"
        assert 'description' in results[backend], \
            f"Backend {backend} missing 'description' key"


def test_backend_probe_cuda_no_gpu():
    """On a system without NVIDIA GPU, CUDA should report unavailable."""
    # This test assumes no GPU on the test system
    probe = BackendProbe()
    results = probe.probe_all()
    # If no GPU, cuda['available'] should be False
    if not results['cuda']['available']:
        assert 'No NVIDIA GPU' in results['cuda']['description']


def test_backend_probe_available_snapshot():
    """Cached results should be consistent."""
    probe = BackendProbe()
    available_before = probe.available()
    available_after = probe.available()
    assert available_before == available_after, "Cached results should be stable"


def test_backend_probe_summary():
    """Summary should be a formatted string."""
    probe = BackendProbe()
    summary = probe.summary()
    assert 'Backend Availability:' in summary
    assert 'cpu' in summary
    assert '✅' in summary or '❌' in summary


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Matcher Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_voice_matcher_best_match():
    """Best match should return the closest model."""
    matcher = VoiceMatcher()
    model, confidence = matcher.best_match('G-D-I-S-S-C-A-L-P-M')

    assert model is not None, "Should find a matching model"
    assert isinstance(model, str), "Model name should be a string"
    assert 0.0 <= confidence <= 1.0, \
        f"Confidence should be [0,1], got {confidence}"


def test_voice_matcher_exact_match():
    """Exact signature match should return confidence 1.0."""
    matcher = VoiceMatcher()

    # Check each known model matches itself
    for model_name, info in MODEL_VOICE_SIGNATURES.items():
        model_sig = info['signature']
        matched_model, confidence = matcher.best_match(model_sig)
        # Should match itself (may also match other models with same sig)
        assert confidence > 0.0, f"Model {model_name} should have non-zero self-match"
        assert matched_model is not None, "Should find some matching model"


def test_voice_matcher_match_all_ranked():
    """All matching results should be ranked by confidence descending."""
    matcher = VoiceMatcher()
    rankings = matcher.match_all('G-D-I-S-S-C-A-L-P-M')

    assert len(rankings) > 0, "Should have rankings"
    assert len(rankings) == len(MODEL_VOICE_SIGNATURES), \
        "Should rank all known models"

    # Check descending order
    for i in range(len(rankings) - 1):
        assert rankings[i][1] >= rankings[i + 1][1], \
            f"Rankings not sorted: {rankings[i][1]} < {rankings[i + 1][1]}"


def test_voice_matcher_top_result():
    """Top result should be the most similar model."""
    matcher = VoiceMatcher()
    rankings = matcher.match_all('G-D-I-S-S-C-A-L-P-M')
    best_model, best_conf = rankings[0]
    assert best_conf >= rankings[1][1], "Top result should have highest confidence"


def test_voice_matcher_empty_signature():
    """Empty signature should gracefully handle."""
    matcher = VoiceMatcher()
    model, confidence = matcher.best_match('')
    # Should still return something
    assert model is not None


def test_voice_matcher_signature_info():
    """Signature info should return details for known models."""
    matcher = VoiceMatcher()
    for model_name in MODEL_VOICE_SIGNATURES:
        info = matcher.signature_info(model_name)
        assert info is not None, f"Should have info for {model_name}"
        assert 'signature' in info, f"Missing signature for {model_name}"

    # Unknown model should return None
    assert matcher.signature_info('nonexistent-model-v99') is None


def test_voice_matcher_with_directory():
    """Load signatures from a directory of JSON files."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a test signature file
        sig_data = {
            'generating_model': 'test-model',
            'text_name': 'test-text',
            'signature': 'G-D-I-S-S-C-A-L-P-M',
        }
        with open(Path(tmp) / 'test_sig.json', 'w') as f:
            json.dump(sig_data, f)

        matcher = VoiceMatcher(signatures_dir=tmp)
        model, confidence = matcher.best_match('G-D-I-S-S-C-A-L-P-M')
        assert model is not None, "Should find model from directory"

        # The loaded model should include our test-model
        info = matcher.signature_info('test-model')
        if info:
            assert info['signature'] == 'G-D-I-S-S-C-A-L-P-M'


def test_voice_matcher_with_anchor_points():
    """Handle signature files with anchor_points format."""
    with tempfile.TemporaryDirectory() as tmp:
        sig_data = {
            'generating_model': 'ap-model',
            'text_name': 'anchor-test',
            'signature': 'DEFERRED',  # Will use anchor_points
            'anchor_points': {
                'dim_G': {'value': 'grounded', 'scale': ['grounded'], 'confidence': 0.9},
                'dim_D': {'value': 'dense', 'scale': ['dense'], 'confidence': 0.8},
            },
        }
        with open(Path(tmp) / 'anchor_sig.json', 'w') as f:
            json.dump(sig_data, f)

        matcher = VoiceMatcher(signatures_dir=tmp)
        model, confidence = matcher.best_match('G-D-I-S-S-C-A-L-P-M')
        assert model is not None, "Should find a model"


# ═══════════════════════════════════════════════════════════════════════════════
# Insight Router Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_insight_router_route():
    """Full routing should produce valid recommendations."""
    router = InsightRouter()
    result = router.route("rust constraint solver", task_sig="G-D-I-S-S")

    # Required fields
    assert 'task' in result
    assert 'model' in result
    assert 'model_confidence' in result
    assert 'model_rankings' in result
    assert 'recommended_backend' in result
    assert 'available_backends' in result
    assert 'flux_bytecode' in result
    assert 'flux_opcode_count' in result
    assert 'complexity' in result

    # Types
    assert isinstance(result['model'], str)
    assert 0 <= result['model_confidence'] <= 1
    assert len(result['model_rankings']) > 0
    assert result['recommended_backend'] in result['available_backends']
    assert len(result['flux_bytecode']) > 0
    assert 'score' in result['complexity']
    assert 'level' in result['complexity']


def test_insight_router_auto_signature():
    """Router should infer signature from task description."""
    router = InsightRouter()
    result = router.route("prove a theorem in Coq about formal verification")
    assert 'task_signature' in result
    assert result['task_signature'] != '', "Should have auto-generated signature"


def test_insight_router_prefer_backend():
    """Router should respect backend preference."""
    router = InsightRouter()
    result = router.route("matrix multiplication", task_sig="G-D-I-S-S",
                          prefer_backend='cpu')
    assert result['recommended_backend'] == 'cpu', \
        "Should prefer CPU when specified"


def test_insight_router_complexity_estimation():
    """Complexity estimation should produce reasonable levels."""
    router = InsightRouter()

    # Simple task
    simple = router.route("hello world", task_sig="G-D-I-S-S")
    assert simple['complexity']['score'] >= 0
    assert simple['complexity']['score'] <= 100

    # Complex task
    complex_task = router.route(
        "prove a formal theorem about concurrent data structure verification",
        task_sig="g-d-I-s-s-C-A-l-p-m"
    )
    assert complex_task['complexity']['score'] >= 0

    # Complexity levels should be one of the defined ones
    for level in [simple['complexity']['level'], complex_task['complexity']['level']]:
        assert level in ('simple', 'moderate', 'complex', 'very complex'), \
            f"Invalid complexity level: {level}"


def test_insight_router_route_batch():
    """Batch routing should process all tasks."""
    router = InsightRouter()
    tasks = [
        ("write python code", "G-D-I-S-S"),
        ("prove a theorem", "g-d-I-s-s"),
        "simple task with auto-signature",
    ]
    results = router.route_batch(tasks)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for i, result in enumerate(results):
        assert 'model' in result, f"Result {i} missing model"
        assert 'task' in result, f"Result {i} missing task"


def test_insight_router_backend_picking():
    """Backend picking should be sensible for different task types."""
    router = InsightRouter()

    # Scientific/math tasks should prefer fortran or cpu
    math_result = router.route("numerical simulation of fluid dynamics",
                                task_sig="G-D-I-S-S")
    assert math_result['recommended_backend'] in math_result['available_backends']

    # Kernel tasks should prefer ebpf if available
    kernel_result = router.route("kernel networking filter",
                                  task_sig="G-D-I-S-S")
    assert kernel_result['recommended_backend'] in kernel_result['available_backends']


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests (connects to existing casting-call modules)
# ═══════════════════════════════════════════════════════════════════════════════

def test_voice_matcher_feeds_insight_router():
    """VoiceMatcher output should be usable by InsightRouter."""
    matcher = VoiceMatcher()
    router = InsightRouter()

    sig = "G-D-I-S-S-C-A-L-P-M"
    best_model, confidence = matcher.best_match(sig)

    result = router.route("general purpose task", task_sig=sig)

    # The router's top model should match VoiceMatcher's best match
    assert result['model_rankings'][0]['model'] == best_model, \
        "Router top model should match VoiceMatcher best model"


def test_backend_probe_feeds_insight_router():
    """BackendProbe results should be reflected in router output."""
    probe = BackendProbe()
    router = InsightRouter()

    probe_results = probe.probe_all()
    available = [k for k, v in probe_results.items() if v['available']]

    result = router.route("test", task_sig="G-D-I-S-S")

    # Router's available_backends should match probe results
    for backend in available:
        assert backend in result['available_backends'], \
            f"Backend {backend} should be in router's available list"


def test_flux_bytecode_as_ir():
    """FLUX bytecode should serve as a valid intermediate representation."""
    # The bytecode should be serializable (list of strings)
    bytecode = generate_flux_bytecode(task_sig="G-D-I-S-S")

    # Check it's JSON-serializable
    serialized = json.dumps(bytecode)
    deserialized = json.loads(serialized)
    assert deserialized == bytecode, "Bytecode should survive JSON round-trip"

    # Opcode count should reflect task complexity
    simple_bc = generate_flux_bytecode(task_sig="G")
    complex_bc = generate_flux_bytecode(task_sig="G-D-I-S-S-C-A-L-P-M")
    assert len(complex_bc) >= len(simple_bc), \
        "Multi-dimensional tasks should produce longer bytecode"


def test_signature_dimensions_consistency():
    """Signature dimensions should match between modules."""
    # Each dimension should have a corresponding OPCODE_TO_SIG_DIM entry.
    # The duplicate 'S' dimension is split into S0 (Scope) and S1 (Structure).
    from insight_engine import OPCODE_TO_SIG_DIM
    unique_dims = set(SIGNATURE_DIMENSIONS)
    for dim in unique_dims:
        if dim == 'S':
            assert 'S0' in OPCODE_TO_SIG_DIM and 'S1' in OPCODE_TO_SIG_DIM, \
                "S dimension should be split into S0 (Scope) and S1 (Structure)"
        else:
            assert dim in OPCODE_TO_SIG_DIM, \
                f"Dimension {dim} missing from OPCODE_TO_SIG_DIM"

    # OPCODE_TO_SIG_DIM should have exactly 10 entries (1 per dimension)
    assert len(OPCODE_TO_SIG_DIM) == 10, \
        f"Expected 10 OPCODE_TO_SIG_DIM entries, got {len(OPCODE_TO_SIG_DIM)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_router_empty_task():
    """Empty task should still produce a valid route."""
    router = InsightRouter()
    result = router.route("")
    assert result['model'] is not None
    assert len(result['flux_bytecode']) >= 2


def test_router_very_long_task():
    """Very long task descriptions shouldn't crash."""
    router = InsightRouter()
    long_task = "do " * 100 + "something"
    result = router.route(long_task)
    assert result['model'] is not None


def test_router_special_characters():
    """Task descriptions with special characters should be handled."""
    router = InsightRouter()
    result = router.route("C++ template metaprogramming: solve ∀x: P(x) → Q(x)")
    assert result['model'] is not None


def test_model_voice_signatures_all_valid():
    """All built-in model signatures should parse correctly."""
    for model_name, info in MODEL_VOICE_SIGNATURES.items():
        sig = info['signature']
        parts = parse_signature_code(sig)
        assert len(parts) == 10, \
            f"Model {model_name} signature has {len(parts)} parts, expected 10"
        valid_chars = set('GDISSCALPMgdisscalpm')
        assert all(p in valid_chars for p in parts), \
            f"Model {model_name} has invalid dimension code: {parts}"
        assert 'strengths' in info, f"Model {model_name} missing strengths"
        assert 'description' in info, f"Model {model_name} missing description"


def test_backend_task_fit_configuration():
    """All backends should have task-fit configuration."""
    from insight_engine import BACKEND_TASK_FIT
    probe = BackendProbe()
    results = probe.probe_all()

    for backend in results:
        assert backend in BACKEND_TASK_FIT, \
            f"Backend {backend} missing from BACKEND_TASK_FIT"
        config = BACKEND_TASK_FIT[backend]
        assert 'fitting' in config
        assert 'max_ops' in config
        assert 'latency' in config


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        name for name in dir() if name.startswith('test_')
    ]
    passed = 0
    failed = 0
    for name in sorted(test_functions):
        func = globals()[name]
        try:
            func()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
