#!/usr/bin/env python3
"""
FM-Oracle1 Insight Engine

Bridge between FM's hardware backends and the casting-call's model voice routing,
using the FLUX 30-opcode constraint subset as the intermediate representation.

Architecture:
    Task description
          │
          ▼
    ┌─────────────────┐     ┌──────────────────────┐
    │  FM's Backends   │     │  Oracle1's Models     │
    │                  │     │                       │
    │ • CPU AVX-512   │     │ • DeepSeek v4-flash   │
    │ • CUDA GPU      │     │ • DeepSeek v4-pro     │
    │ • FPGA/RTL      │     │ • GLM-5.1             │
    │ • eBPF          │     │ • Seed-2.0-mini       │
    │ • WebGPU/Vulkan │     │ • Kimi K2.5           │
    │ • Fortran       │     │ • Hermes-405B         │
    │ • Coq formal    │     │ • Casting-call DB     │
    └────────┬────────┘     └──────────┬─────────────┘
             │                         │
             └──────────┬──────────────┘
                        │
                   ┌────▼────┐
                   │  FLUX   │
                   │ 30-     │
                   │ opcode  │
                   │ subset  │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Insight │
                   │ Engine  │
                   └─────────┘

Usage:
    from insight_engine import BackendProbe, VoiceMatcher, InsightRouter

    router = InsightRouter()
    result = router.route("rust constraint solver", task_sig="G-D-I-S-S")
    # → model: deepseek/deepseek-v4-flash, backend: cpu, confidence: 0.85
"""

import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════════
# FLUX 30-Opcode Subset
# ═══════════════════════════════════════════════════════════════════════════════

# The FLUX 30-opcode constraint subset — canonical order and semantics.
# This is the intermediate representation between backends and models.
FLUX_OPCODES = [
    # Arithmetic (5)
    'ADD',   # 0x00
    'SUB',   # 0x01
    'MUL',   # 0x02
    'DIV',   # 0x03
    'MOD',   # 0x04

    # Logic (4)
    'AND',   # 0x04
    'OR',    # 0x05
    'XOR',   # 0x06
    'NOT',   # 0x07

    # Comparison (5)
    'EQ',    # 0x08
    'LT',    # 0x09
    'GT',    # 0x0A
    'LTE',   # 0x0B
    'GTE',   # 0x0C

    # Memory (2)
    'LOAD',  # 0x0D
    'STORE', # 0x0E

    # Control (5)
    'JMP',   # 0x0F
    'JZ',    # 0x10
    'JNZ',   # 0x11
    'CALL',  # 0x12
    'RET',   # 0x13

    # Data movement (2)
    'MOV',   # 0x14
    'SWAP',  # 0x15

    # Type operations (3)
    'CAST',  # 0x16
    'PACK',  # 0x17
    'UNPACK',# 0x18

    # Special (5) — 30 core + 1 constraint = 31 total
    'NOP',   # 0x19
    'HALT',  # 0x1A
    'SIG',   # 0x1B — FLUX-specific constraint op (signature check)
    'DEBUG', # 0x1C
]

# Map opcodes to their hex codes
FLUX_OPCODE_MAP = {op: hex(i) for i, op in enumerate(FLUX_OPCODES)}

# Opcode categories for routing decisions
OPCODE_CATEGORIES = {
    'compute': {'ADD', 'SUB', 'MUL', 'DIV', 'MOD'},
    'logical': {'AND', 'OR', 'XOR', 'NOT'},
    'compare': {'EQ', 'LT', 'GT', 'LTE', 'GTE'},
    'memory':  {'LOAD', 'STORE'},
    'control': {'JMP', 'JZ', 'JNZ', 'CALL', 'RET'},
    'data':    {'MOV', 'SWAP'},
    'type':    {'CAST', 'PACK', 'UNPACK'},
    'special': {'NOP', 'HALT', 'SIG', 'DEBUG'},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Task-to-FLUX Signature Encoding
# ═══════════════════════════════════════════════════════════════════════════════

# The 10-dimension anchor signature used in casting-call, each dimension
# maps to a letter code. Used for Hamming-distance comparison.
SIGNATURE_DIMENSIONS = [
    'G',  # Grounding — how concrete/abstract the output is
    'D',  # Density — information per token
    'I',  # Instruction-following — adherence to exact specification
    'S',  # Scope — breadth of context considered
    'S',  # Structure — degree of output structure
    'C',  # Creativity — novelty vs faithfulness
    'A',  # Alignment — constraint satisfaction strength
    'L',  # Latency — response speed preference
    'P',  # Precision — numerical/formal accuracy needed
    'M',  # Modality — single vs multi-modal
]

# Mapping from opcode categories to typical signature dimensions.
# Used when generating FLUX bytecode from a task description.
OPCODE_TO_SIG_DIM = {
    # Use the canonical dimension list order. Duplicate 'S' entries are
    # disambiguated by index: S0 (Scope) and S1 (Structure).
    'G': {'ADD', 'SUB', 'CAST'},           # Grounding — arithmetic + type ops
    'D': {'LOAD', 'STORE', 'PACK'},        # Density — memory operations
    'I': {'EQ', 'SIG'},                    # Instruction-following — comparison + SIG
    'S0': {'JMP', 'JZ', 'JNZ', 'CALL'},   # Scope — control flow
    'S1': {'NOP', 'RET', 'MOV'},           # Structure — data/special ops
    'C': {'XOR', 'OR', 'NOT'},             # Creativity — logical ops
    'A': {'LT', 'GT', 'LTE', 'GTE'},       # Alignment — comparison
    'L': {'NOP', 'HALT'},                  # Latency — special ops
    'P': {'MUL', 'DIV', 'MOD'},            # Precision — arithmetic
    'M': {'PACK', 'UNPACK', 'CAST'},       # Modality — type ops
}


def parse_signature_code(code: str) -> List[str]:
    """
    Parse a hyphen-separated signature code into its dimension components.

    Example: 'G-D-I-S-S' → ['G', 'D', 'I', 'S', 'S']
    """
    if not code or not isinstance(code, str):
        return []
    return [c.strip() for c in code.split('-') if c.strip()]


def hamming_distance(sig_a: str, sig_b: str) -> int:
    """
    Compute Hamming distance between two signature codes.
    Codes are compared position-by-position after parsing.

    Example: hamming_distance('G-D-I-S-S', 'G-D-I-S-C') → 1
    """
    if not sig_a or not sig_b:
        return max(len(sig_a), len(sig_b)) if (sig_a or sig_b) else 0

    parts_a = parse_signature_code(sig_a)
    parts_b = parse_signature_code(sig_b)

    # Pad shorter code with empty strings
    max_len = max(len(parts_a), len(parts_b))
    parts_a = parts_a + [''] * (max_len - len(parts_a))
    parts_b = parts_b + [''] * (max_len - len(parts_b))

    return sum(1 for a, b in zip(parts_a, parts_b) if a != b)


def encode_task_to_signature(task: str) -> str:
    """
    Encode a free-form task description into a FLUX anchor-point signature code.

    Uses keyword heuristics to assign each of the 10 dimensions a value:
      0 (lower) → first letter of the scale, e.g. 'G' for minimal grounding
      1 (higher) → second letter of the scale, e.g. 'D' for heavy density

    Returns a hyphen-separated code like 'G-D-I-S-S-C-A-L-P-M'.
    """
    task_lower = task.lower()

    # Each dimension has high-end and low-end keywords.
    # DIMENSION_SPEC is an ordered list so duplicate 'S' keys are disambiguated
    # by position: index 3 (Scope → S0 in opcode map), index 4 (Structure → S1).
    DIMENSION_SPEC = [
        ('G', ('concrete', 'specific', 'practical', 'real-world'),
              ('abstract', 'theoretical', 'philosophical', 'speculative')),
        ('D', ('dense', 'concise', 'compressed', 'terse'),
              ('verbose', 'detailed', 'expansive', 'elaborate')),
        ('I', ('exact', 'precise', 'strict', 'rules', 'spec'),
              ('creative', 'open-ended', 'loose', 'interpret')),
        ('S', ('broad', 'comprehensive', 'system', 'holistic'),
              ('narrow', 'focused', 'single', 'specific')),            # Scope
        ('S', ('structured', 'organized', 'outlined', 'formatted'),
              ('freeform', 'prose', 'stream', 'unorganized')),          # Structure
        ('C', ('creative', 'novel', 'surprising', 'innovative'),
              ('faithful', 'accurate', 'literal', 'strict')),
        ('A', ('constrained', 'aligned', 'safe', 'guarded'),
              ('open', 'unconstrained', 'free', 'unrestricted')),
        ('L', ('fast', 'quick', 'urgent', 'realtime'),
              ('deep', 'thorough', 'reasoning', 'careful')),
        ('P', ('precise', 'exact', 'numerical', 'formal'),
              ('approximate', 'rough', 'ballpark', 'estimate')),
        ('M', ('multimodal', 'image', 'audio', 'video'),
              ('text', 'single', 'pure', 'code')),
    ]

    parts = []
    for dim_key, high_kws, low_kws in DIMENSION_SPEC:
        high_count = sum(1 for kw in high_kws if kw in task_lower)
        low_count = sum(1 for kw in low_kws if kw in task_lower)

        if high_count > low_count:
            parts.append(dim_key)       # High end → uppercase
        elif low_count > high_count:
            parts.append(dim_key.lower())  # Low end → lowercase
        else:
            parts.append(dim_key.lower())  # Default to lowercase

    # Convert to canonical form: uppercase = high end, lowercase = low end
    # The casing reflects whether the high-end keywords outmatched the low-end.
    sig = '-'.join(parts)
    return sig


def generate_flux_bytecode(task: str = "", task_sig: str = "") -> List[str]:
    """
    Generate a valid FLUX 30-opcode bytecode sequence from a task description
    and/or task signature.

    The bytecode encodes the computational requirements of the task:
    - Each opcode is chosen based on which signature dimensions are active
    - The sequence length reflects task complexity
    - The opcode composition reflects the task's character

    Returns a list of opcode strings (e.g. ['ADD', 'MOV', 'EQ', ...])
    """
    # Parse the signature if provided, otherwise infer from task
    if not task_sig and task:
        task_sig = encode_task_to_signature(task)

    signature_parts = parse_signature_code(task_sig) if task_sig else []

    # Map each active signature dimension to its associated opcodes.
    # For duplicate 'S' dimensions (Scope=index 3, Structure=index 4),
    # we disambiguate by position: S0 for Scope, S1 for Structure.
    selected_opcodes = []
    for idx, dim in enumerate(signature_parts):
        # Map duplicate S dimensions by position: first S → S0, second S → S1
        if dim == 'S':
            # Check how many S's we've seen so far in the full signature
            s_count = sum(1 for d in signature_parts[:idx] if d == 'S')
            lookup_key = f'S{s_count}'
        else:
            lookup_key = dim

        if lookup_key in OPCODE_TO_SIG_DIM:
            # Pick one opcode from each dimension's associated set
            ops = list(OPCODE_TO_SIG_DIM[lookup_key])
            selected_opcodes.append(ops[len(selected_opcodes) % len(ops)])

    # Build a canonical bytecode sequence:
    # 1. Data movement (setup)
    # 2. Core computation (selected ops)
    # 3. Comparison/verification (SIG at end)
    bytecode = ['MOV', 'NOP']  # Prologue

    # Interleave selected ops with data movement
    for i, op in enumerate(selected_opcodes):
        if op not in {'LOAD', 'STORE'}:
            bytecode.append('LOAD')
            bytecode.append(op)
            bytecode.append('STORE')
        else:
            bytecode.append(op)

    # Always include a SIG op as the constraint verifier
    bytecode.append('SIG')
    bytecode.append('HALT')  # Epilogue

    return bytecode


# ═══════════════════════════════════════════════════════════════════════════════
# Backend Probe
# ═══════════════════════════════════════════════════════════════════════════════

class BackendProbe:
    """
    Discover available FLUX hardware backends on the current system.

    Checks for:
      - CPU (always available — reference implementation)
      - CUDA (NVIDIA GPU)
      - FPGA (Yosys/NextPNR toolchain)
      - eBPF (Linux kernel bytecode)
      - WebGPU (via node or browser)
      - Vulkan (compute shaders)
      - Fortran (scientific compute)
      - Coq (formal verification)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._cache = None

    def probe_all(self) -> Dict[str, dict]:
        """
        Check which FLUX hardware backends are available on this system.

        Returns a dict mapping backend name → {'available': bool, 'detail': str, ...}
        """
        results = {}

        # CPU — always available (reference implementation via flux_runtime_arm.c)
        results['cpu'] = self._probe_cpu()

        # CUDA — NVIDIA GPU via nvidia-smi or CuPy
        results['cuda'] = self._probe_cuda()

        # FPGA — Yosys/NextPNR toolchain
        results['fpga'] = self._probe_fpga()

        # eBPF — Linux kernel BPF
        results['ebpf'] = self._probe_ebpf()

        # WebGPU — via node or browser detection
        results['webgpu'] = self._probe_webgpu()

        # Vulkan — compute shaders via vulkaninfo
        results['vulkan'] = self._probe_vulkan()

        # Fortran — scientific compute
        results['fortran'] = self._probe_fortran()

        # Coq — formal verification
        results['coq'] = self._probe_coq()

        self._cache = results
        return results

    def _log(self, msg: str):
        if self.verbose:
            print(f"[BackendProbe] {msg}", file=sys.stderr)

    def _probe_cpu(self) -> dict:
        """CPU is always available as the reference implementation."""
        info = {
            'available': True,
            'backend': 'flux_runtime_arm.c',
            'description': 'CPU reference implementation (always available)',
        }

        # Check for AVX-512 support (optional detail)
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()
            if 'avx512' in cpuinfo.lower():
                info['features'] = ['avx512']
                info['description'] = 'CPU with AVX-512 support'
        except (FileNotFoundError, IOError):
            pass

        # Check for ARM NEON
        try:
            import platform
            if 'aarch64' in platform.machine():
                info['features'] = info.get('features', []) + ['neon']
                info['description'] = 'CPU with ARM NEON'
        except Exception:
            pass

        return info

    def _probe_cuda(self) -> dict:
        """Check for NVIDIA GPU via nvidia-smi."""
        # Primary check: nvidia-smi
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            try:
                result = subprocess.run(
                    [nvidia_smi, '--query-gpu=name,driver_version,memory.total',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(', ')
                    return {
                        'available': True,
                        'backend': 'cuda',
                        'gpu_name': parts[0] if len(parts) > 0 else 'unknown',
                        'driver_version': parts[1] if len(parts) > 1 else 'unknown',
                        'memory_total': parts[2] if len(parts) > 2 else 'unknown',
                        'description': f"CUDA GPU ({parts[0] if len(parts) > 0 else 'unknown'})",
                    }
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
                pass

        # Fallback: check for CuPy
        try:
            import cupy as cp
            if cp.cuda.is_available():
                device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                return {
                    'available': True,
                    'backend': 'cupy',
                    'gpu_name': device_name,
                    'description': f"CUDA GPU via CuPy ({device_name})",
                }
        except (ImportError, Exception):
            pass

        # Fallback: check for PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return {
                    'available': True,
                    'backend': 'torch_cuda',
                    'gpu_name': device_name,
                    'description': f"CUDA GPU via PyTorch ({device_name})",
                }
        except (ImportError, Exception):
            pass

        return {
            'available': False,
            'backend': None,
            'description': 'No NVIDIA GPU detected',
        }

    def _probe_fpga(self) -> dict:
        """Check for FPGA toolchain (Yosys/NextPNR)."""
        tools = {
            'yosys': shutil.which('yosys'),
            'nextpnr': shutil.which('nextpnr'),
            'verilog': shutil.which('iverilog'),
        }

        available_tools = {k: v for k, v in tools.items() if v}
        return {
            'available': len(available_tools) > 0,
            'tools': available_tools,
            'description': f"FPGA tools: {', '.join(available_tools.keys()) if available_tools else 'none'}",
        }

    def _probe_ebpf(self) -> dict:
        """Check for eBPF toolchain (bpftool, clang with bpf target)."""
        bpftool = shutil.which('bpftool')

        info = {
            'available': bpftool is not None,
            'description': '',
        }

        if bpftool:
            try:
                result = subprocess.run(
                    [bpftool, '--version'],
                    capture_output=True, text=True, timeout=3
                )
                info['version'] = result.stdout.strip() or result.stderr.strip()
                info['description'] = f"eBPF available ({info['version']})"
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                info['description'] = 'eBPF available (bpftool found)'

            # Check kernel support
            try:
                with open('/sys/kernel/btf/vmlinux', 'rb') as _:
                    info['btf'] = True
            except (FileNotFoundError, IOError):
                info['btf'] = False
        else:
            info['description'] = 'No eBPF toolchain detected'

        return info

    def _probe_webgpu(self) -> dict:
        """Check for WebGPU via node or browser."""
        webgpu = None

        # Check for Node.js with WebGPU support
        node = shutil.which('node')
        if node:
            try:
                result = subprocess.run(
                    [node, '-e', 'console.log(typeof WebGPU !== "undefined")'],
                    capture_output=True, text=True, timeout=3
                )
                if 'true' in result.stdout.lower():
                    webgpu = 'node'
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        # Check for node-webgpu package
        if not webgpu and node:
            try:
                result = subprocess.run(
                    [node, '-e',
                     'try{require("@webgpu/types");console.log("types")}catch(e){}'
                     'try{require("webgpu");console.log("webgpu")}catch(e){}'],
                    capture_output=True, text=True, timeout=3
                )
                if result.stdout.strip():
                    webgpu = 'node-package'
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        # Check for Chrome-based browser
        browsers = ['google-chrome', 'chromium-browser', 'chromium']
        browser_path = None
        for b in browsers:
            path = shutil.which(b)
            if path:
                browser_path = path
                break

        return {
            'available': webgpu is not None or browser_path is not None,
            'runtime': webgpu or ('browser' if browser_path else None),
            'browser': browser_path,
            'description': f"WebGPU via {webgpu or (browser_path or 'not detected')}",
        }

    def _probe_vulkan(self) -> dict:
        """Check for Vulkan compute support via vulkaninfo."""
        vulkaninfo = shutil.which('vulkaninfo')
        if not vulkaninfo:
            # Also check for Vulkan SDK loader
            vulkan_loader = shutil.which('libvulkan.so.1') or \
                            shutil.which('vulkan-loader')
            return {
                'available': False,
                'description': 'No Vulkan toolchain detected',
            }

        try:
            result = subprocess.run(
                [vulkaninfo, '--summary'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and 'Vulkan' in (result.stdout + result.stderr):
                # Extract GPU info
                gpu_match = re.search(r'GPU[^:]*:\s*([^\n]+)',
                                      result.stdout + result.stderr)
                gpu_name = gpu_match.group(1).strip() if gpu_match else 'unknown'
                return {
                    'available': True,
                    'gpu_name': gpu_name,
                    'description': f"Vulkan compute ({gpu_name})",
                }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return {
            'available': False,
            'description': 'Vulkan tool found but no GPU detected',
        }

    def _probe_fortran(self) -> dict:
        """Check for Fortran compiler (scientific compute)."""
        gfortran = shutil.which('gfortran')
        if gfortran:
            try:
                result = subprocess.run(
                    [gfortran, '--version'],
                    capture_output=True, text=True, timeout=3
                )
                version_line = result.stdout.split('\n')[0] if result.stdout else 'unknown'
                return {
                    'available': True,
                    'compiler': gfortran,
                    'version': version_line,
                    'description': f"Fortran ({version_line})",
                }
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        return {
            'available': False,
            'description': 'No Fortran compiler detected',
        }

    def _probe_coq(self) -> dict:
        """Check for Coq proof assistant (formal verification)."""
        coqc = shutil.which('coqc')
        if coqc:
            try:
                result = subprocess.run(
                    [coqc, '--version'],
                    capture_output=True, text=True, timeout=3
                )
                version_line = result.stdout.split('\n')[0] if result.stdout else 'unknown'
                return {
                    'available': True,
                    'compiler': coqc,
                    'version': version_line,
                    'description': f"Coq formal verification ({version_line})",
                }
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        return {
            'available': False,
            'description': 'No Coq installation detected',
        }

    def available(self) -> List[str]:
        """Return list of available backend names."""
        if self._cache is None:
            self.probe_all()
        return [k for k, v in self._cache.items() if v['available']]

    def summary(self) -> str:
        """Return a human-readable summary of available backends."""
        if self._cache is None:
            self.probe_all()
        lines = ['Backend Availability:']
        for name, info in self._cache.items():
            status = '✅' if info['available'] else '❌'
            lines.append(f"  {status} {name}: {info['description']}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Matcher
# ═══════════════════════════════════════════════════════════════════════════════

# Default known model voice signatures (from casting-call database).
# Each model has a canonical 10-dimension signature code.
MODEL_VOICE_SIGNATURES = {
    'deepseek/deepseek-v4-flash': {
        'signature': 'G-D-I-S-S-c-A-L-P-m',
        'description': 'Fast, precise, instruction-following code model',
        'strengths': ['code', 'analysis', 'review', 'iteration'],
    },
    'deepseek/deepseek-v4-pro': {
        'signature': 'g-d-I-S-s-c-A-l-P-m',
        'description': 'Deep reasoning, mathematical proofs, formal verification',
        'strengths': ['math', 'reasoning', 'proof', 'constraint theory'],
    },
    'zai/glm-5.1': {
        'signature': 'G-D-I-S-s-c-a-L-p-m',
        'description': 'Expert agent, general-purpose, strong instruction following',
        'strengths': ['planning', 'architecture', 'coordination', 'general'],
    },
    'zai/glm-5-turbo': {
        'signature': 'G-d-I-s-s-c-A-l-P-m',
        'description': 'Fast daily driver, balanced performance and speed',
        'strengths': ['implementation', 'refactoring', 'daily driver'],
    },
    'moonshot/kimi-k2.5': {
        'signature': 'G-d-I-S-S-C-A-l-p-m',
        'description': 'Reasoning + creative combined, deep research',
        'strengths': ['research', 'swarm analysis', 'creative+reasoning'],
    },
    'deepinfra/seed-2.0-mini': {
        'signature': 'g-d-i-s-s-C-a-l-p-M',
        'description': 'Divergent thinker, creative breadth, cheap and fast',
        'strengths': ['creative', 'divergent thinking', 'exploration'],
    },
    'deepinfra/seed-2.0-mini-creative': {
        'signature': 'g-d-i-s-s-C-a-l-p-M',
        'description': 'Same model at high temperature, maximum creative breadth',
        'strengths': ['brainstorming', 'options generation', 'naming'],
    },
    'deepinfra/nemotron-3-reasoning': {
        'signature': 'g-d-I-s-s-c-A-l-p-m',
        'description': 'Fallback reasoning model on DeepInfra',
        'strengths': ['reasoning', 'analysis', 'fallback'],
    },
    'hermes/hermes-405b': {
        'signature': 'G-D-I-S-S-c-a-L-P-m',
        'description': 'Large general-purpose model, broad scope, structured output',
        'strengths': ['general', 'structured output', 'broad tasks'],
    },
}


class VoiceMatcher:
    """
    Match a task description to the best model voice using the casting-call
    signature database.

    Compares the task's signature code against each model's stored signature
    using Hamming distance. The closest match is the recommended model voice.
    """

    def __init__(self, signatures_dir: Optional[str] = None):
        """
        Initialize voice matcher.

        Args:
            signatures_dir: Optional path to casting-call signature JSON files.
                            If provided, loads model voice signatures from files.
                            Otherwise uses built-in MODEL_VOICE_SIGNATURES.
        """
        self.signatures_dir = signatures_dir
        self._signatures = None

    def _load_signatures(self) -> Dict[str, dict]:
        """Load voice signatures, from files or built-in defaults."""
        if self._signatures is not None:
            return self._signatures

        if self.signatures_dir and os.path.isdir(self.signatures_dir):
            sigs = self._load_from_directory(self.signatures_dir)
            if sigs:
                self._signatures = sigs
                return sigs

        self._signatures = MODEL_VOICE_SIGNATURES
        return self._signatures

    def _load_from_directory(self, sig_dir: str) -> Dict[str, dict]:
        """Load voice signatures from JSON files in a directory."""
        sig_path = Path(sig_dir)
        if not sig_path.is_dir():
            return {}

        signatures = {}
        for f in sorted(sig_path.glob('*.json')):
            try:
                with open(f) as fh:
                    data = json.load(fh)

                model_name = data.get('generating_model', f.stem)
                sig_raw = data.get('signature', '')

                # The signature might be in anchor_points or as a direct code
                if isinstance(sig_raw, str) and '-' in sig_raw:
                    # It's a hyphen-separated code
                    pass
                elif isinstance(sig_raw, dict):
                    # Try to extract signature from anchor features
                    ap = data.get('anchor_points', {})
                    if ap:
                        sig_raw = self._derive_signature_from_anchor_points(ap)
                    else:
                        sig_raw = encode_task_to_signature(
                            data.get('text_name', f.stem)
                        )
                else:
                    continue

                signatures[model_name] = {
                    'signature': sig_raw,
                    'source': str(f),
                    'text_name': data.get('text_name', f.stem),
                }
            except (json.JSONDecodeError, KeyError, IOError) as e:
                warnings.warn(f"Could not load signature from {f}: {e}")

        return signatures

    def _derive_signature_from_anchor_points(self, ap: dict) -> str:
        """Derive a signature code from anchor points dictionary."""
        parts = []
        for dim_key in SIGNATURE_DIMENSIONS:
            # Look for a matching anchor point
            found = False
            for ap_key, ap_val in ap.items():
                if isinstance(ap_val, dict) and 'value' in ap_val:
                    val = str(ap_val['value'])
                    confidence = ap_val.get('confidence', 0.5)
                    if dim_key in val.upper() or dim_key.lower() in ap_key.lower():
                        parts.append(dim_key if confidence > 0.5 else dim_key.lower())
                        found = True
                        break
            if not found:
                parts.append(dim_key.lower())  # Default to low end

        return '-'.join(parts).upper()

    def best_match(self, task_sig: str) -> Tuple[Optional[str], float]:
        """
        Find the model with the closest voice signature to the requested one.

        Args:
            task_sig: A hyphen-separated signature code like 'G-D-I-S-S'

        Returns:
            (model_name, confidence) where confidence is 1.0 - normalized
            Hamming distance. Returns (None, 0.0) if no models are loaded.
        """
        signatures = self._load_signatures()
        if not signatures:
            return None, 0.0

        best_model = None
        best_dist = float('inf')

        for model_name, info in signatures.items():
            model_sig = info.get('signature', '')
            if not model_sig:
                continue

            dist = hamming_distance(task_sig, model_sig)
            if dist < best_dist:
                best_dist = dist
                best_model = model_name

        if best_model is None:
            return None, 0.0

        # Normalize confidence: max distance is the number of dimensions (10)
        max_len_task = len(parse_signature_code(task_sig))
        max_len_model = max(
            (len(parse_signature_code(s.get('signature', '')))
             for s in signatures.values()
             if isinstance(s.get('signature', ''), str)),
            default=max_len_task
        )
        max_possible = max(max_len_task, max_len_model)
        confidence = 1.0 - (best_dist / max(1, max_possible))
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return best_model, confidence

    def match_all(self, task_sig: str) -> List[Tuple[str, float]]:
        """
        Return all models ranked by similarity.

        Args:
            task_sig: A hyphen-separated signature code

        Returns:
            List of (model_name, confidence) sorted by descending confidence
        """
        signatures = self._load_signatures()
        if not signatures:
            return []

        results = []
        max_possible = len(parse_signature_code(task_sig)) or 10

        for model_name, info in signatures.items():
            model_sig = info.get('signature', '')
            if not model_sig:
                continue
            dist = hamming_distance(task_sig, model_sig)
            # Use the longer of the two codes for normalization
            max_dim = max(
                len(parse_signature_code(task_sig)),
                len(parse_signature_code(model_sig)),
                1
            )
            confidence = 1.0 - (dist / max_dim)
            confidence = max(0.0, min(1.0, confidence))
            results.append((model_name, confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def signature_info(self, model_name: str) -> Optional[dict]:
        """Get the voice signature info for a specific model."""
        signatures = self._load_signatures()
        return signatures.get(model_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Insight Router
# ═══════════════════════════════════════════════════════════════════════════════

# Backend-to-task-type mapping for recommended backend selection
BACKEND_TASK_FIT = {
    'cpu':     {'fitting': ['small', 'simple', 'general', 'reference'],
                'max_ops': 100, 'latency': 'low'},
    'cuda':    {'fitting': ['large', 'parallel', 'matrix', 'gpu'],
                'max_ops': 100000, 'latency': 'medium'},
    'fpga':    {'fitting': ['deterministic', 'pipeline', 'realtime', 'fixed'],
                'max_ops': 50000, 'latency': 'low'},
    'ebpf':    {'fitting': ['kernel', 'networking', 'tracing', 'filter'],
                'max_ops': 4096, 'latency': 'very low'},
    'webgpu':  {'fitting': ['browser', 'visualization', 'web', 'canvas'],
                'max_ops': 10000, 'latency': 'medium'},
    'vulkan':  {'fitting': ['compute', 'shader', 'graphics', 'parallel'],
                'max_ops': 50000, 'latency': 'low'},
    'fortran': {'fitting': ['scientific', 'numerical', 'simulation', 'math'],
                'max_ops': 100000, 'latency': 'high'},
    'coq':     {'fitting': ['proof', 'verification', 'formal', 'theorem'],
                'max_ops': 1000, 'latency': 'high'},
}


class InsightRouter:
    """
    Combine backend probing + voice matching to route a task to the
    optimal combination of hardware backend and AI model voice.

    The router:
    1. Probes available hardware backends
    2. Matches the task signature to the best model voice
    3. Recommends the optimal backend for the task
    4. Generates FLUX 30-opcode bytecode as the intermediate representation
    """

    def __init__(self,
                 signatures_dir: Optional[str] = None,
                 verbose: bool = False):
        self.backend_probe = BackendProbe(verbose=verbose)
        self.voice_matcher = VoiceMatcher(signatures_dir=signatures_dir)
        self.verbose = verbose

    def route(self,
              task: str,
              task_sig: Optional[str] = None,
              prefer_backend: Optional[str] = None) -> Dict:
        """
        Route a task to the optimal backend + model combination.

        Args:
            task: Free-form task description (e.g. "rust constraint solver")
            task_sig: Optional pre-computed signature code (e.g. "G-D-I-S-S").
                      If not provided, inferred from the task description.
            prefer_backend: Optional preferred backend override.

        Returns:
            dict with routing recommendation
        """
        # Step 1: Encode task to signature if not provided
        if not task_sig:
            task_sig = encode_task_to_signature(task)

        # Step 2: Probe available backends
        backends = self.backend_probe.probe_all()

        # Step 3: Find best model voice match
        best_model, model_confidence = self.voice_matcher.best_match(task_sig)
        all_matches = self.voice_matcher.match_all(task_sig)

        # Step 4: Generate FLUX bytecode
        flux_bytecode = generate_flux_bytecode(task, task_sig)

        # Step 5: Pick best backend
        available_backends = {k: v for k, v in backends.items() if v['available']}

        if prefer_backend and prefer_backend in available_backends:
            recommended_backend = prefer_backend
        else:
            recommended_backend = self._pick_backend(available_backends, task, task_sig)

        # Step 6: Estimate complexity from bytecode
        complexity = self._estimate_complexity(flux_bytecode, task, task_sig)

        return {
            'task': task,
            'task_signature': task_sig,
            'model': best_model or 'unknown',
            'model_confidence': round(model_confidence, 4),
            'model_rankings': [
                {'model': m, 'confidence': round(c, 4)}
                for m, c in all_matches[:5]  # Top 5
            ],
            'recommended_backend': recommended_backend,
            'available_backends': list(available_backends.keys()),
            'backend_details': available_backends,
            'flux_bytecode': flux_bytecode,
            'flux_opcode_count': len(flux_bytecode),
            'complexity': complexity,
        }

    def _pick_backend(self,
                      available_backends: Dict[str, dict],
                      task: str,
                      task_sig: str) -> str:
        """
        Pick the best available backend for a given task.

        Uses keyword matching against backend task-fit profiles.
        Falls back to 'cpu' if nothing matches well.
        """
        if 'cpu' not in available_backends:
            # Should never happen, but just in case
            available = list(available_backends.keys())
            return available[0] if available else 'cpu'

        task_lower = task.lower()

        # Score each available backend
        scores = {}
        for backend_name, backend_info in BACKEND_TASK_FIT.items():
            if backend_name not in available_backends:
                continue

            score = 0
            # Check keyword matches
            for kw in backend_info['fitting']:
                if kw in task_lower:
                    score += 2

            # Check signature-based characteristics
            sig_parts = parse_signature_code(task_sig)
            if sig_parts:
                # Tasks with high precision (P) → cpu or fortran
                if 'P' in sig_parts and backend_name in ('cpu', 'fortran'):
                    score += 1
                # Tasks with high modality (M) → webgpu or vulkan
                if 'M' in sig_parts and backend_name in ('webgpu', 'vulkan'):
                    score += 1
                # Tasks with high instruction-following (I) → cpu or coq
                if 'I' in sig_parts and backend_name in ('cpu', 'coq'):
                    score += 1

            scores[backend_name] = score

        if not scores:
            return 'cpu'

        # Return the backend with the highest score
        return max(scores, key=scores.get)

    def _estimate_complexity(self,
                             bytecode: List[str],
                             task: str,
                             task_sig: str) -> Dict:
        """Estimate task complexity from bytecode and task description."""
        opcode_count = len(bytecode)

        # Count by category
        categories = {}
        for op in bytecode:
            for cat_name, cat_ops in OPCODE_CATEGORIES.items():
                if op in cat_ops:
                    categories[cat_name] = categories.get(cat_name, 0) + 1
                    break

        sig_parts = parse_signature_code(task_sig)

        # Rough complexity score
        base_score = min(opcode_count * 10, 100)
        if len(sig_parts) >= 8:
            base_score += 20  # Multi-dimensional tasks are complex
        if any(kw in task.lower() for kw in ['proof', 'verification', 'formal']):
            base_score += 30
        if any(kw in task.lower() for kw in ['large', 'massive', 'huge']):
            base_score += 20

        complexity_score = min(base_score, 100)

        if complexity_score < 30:
            level = 'simple'
        elif complexity_score < 60:
            level = 'moderate'
        elif complexity_score < 85:
            level = 'complex'
        else:
            level = 'very complex'

        return {
            'score': complexity_score,
            'level': level,
            'opcode_count': opcode_count,
            'opcode_categories': categories,
        }

    def route_batch(self, tasks: List[Union[str, Tuple[str, str]]]) -> List[Dict]:
        """
        Route multiple tasks in batch.

        Args:
            tasks: List of tasks. Each entry is either:
                   - A string (task description)
                   - A tuple of (task_description, task_signature)

        Returns:
            List of routing results
        """
        results = []
        for entry in tasks:
            if isinstance(entry, tuple):
                task, task_sig = entry
            else:
                task = entry
                task_sig = None
            results.append(self.route(task, task_sig=task_sig))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def format_bytecode(bytecode: List[str]) -> str:
    """Format FLUX bytecode as a human-readable hex dump."""
    lines = ['FLUX Bytecode:']
    for i, op in enumerate(bytecode):
        hex_code = FLUX_OPCODE_MAP.get(op, '??')
        lines.append(f"  [{i:3d}] {hex_code:>4s}  {op}")
        if i > 0 and (i + 1) % 16 == 0:
            lines.append('')  # Blank line every 16 ops
    return '\n'.join(lines)
