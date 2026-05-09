# Casting-Call GPU 🚀


## Meta

**Domain:** other
**Depends on:** —
**Depended by:** —
**Implements:** GPU-native engine for anchor-point signature matrices, voice spline interpolatio...
**Related:** —


**GPU-native engine for anchor-point signature mathematics.**

The mathematical engine behind the Casting-Call theory. Computes signature distance
matrices, voice spline interpolation, and model clustering — with GPU acceleration
and graceful CPU fallback.

```
signature.json ──→ cast_gpu ──→ distance_matrix.json
                                    → cluster_assignments.json
                                    → interpolated_signature
```

---

## Installation

```bash
git clone https://github.com/SuperInstance/casting-call-gpu.git
cd casting-call-gpu
pip install -r requirements.txt

# Optional: GPU acceleration
pip install cupy-cuda12x  # NVIDIA GPU
# or
pip install torch         # PyTorch (CPU/GPU/MPS)
```

---

## CLI Usage

```bash
# Check available compute device
python cast_gpu.py info

# Compute signature distance matrix
python cast_gpu.py matrix /tmp/casting-call/signatures/ --metric cosine -o matrix.json

# Cluster signatures by model
python cast_gpu.py cluster /tmp/casting-call/signatures/ --k 4

# Full pipeline
python cast_gpu.py matrix signatures/ && python cast_gpu.py cluster signatures/
```

### Example: Signature Matrix

```bash
$ python cast_gpu.py matrix /tmp/casting-call/signatures/
Loaded 15 signatures with 22 features each
Using: GPU (NVIDIA A10G)
Matrix saved to signature_matrix.json
{
  "shape": [15, 15],
  "min_distance": 0.02,
  "max_distance": 0.89,
  "mean_distance": 0.47,
  "most_similar": {"a": "eileen.json", "b": "charter.json", "distance": 0.02},
  "least_similar": {"a": "1000years.json", "b": "readme.json", "distance": 0.89}
}
```

---

## Python API

### Signature Matrix

```python
from cast_gpu import SignatureMatrix, load_signatures

# Load signatures from directory
sigs, meta = load_signatures('path/to/signatures/')
# sigs.shape: (15, 22) — 15 signatures, 22 features each

# Compute distance matrix
mat = SignatureMatrix(device='auto')  # auto-detects GPU
distances = mat.compute(sigs, metric='cosine')

# Analyze results
summary = mat.summarize(distances, meta)
print(f"Mean distance: {summary['mean_distance']:.2f}")
print(f"Most similar: {summary['most_similar']}")
```

### Voice Spline Interpolation

```python
from cast_gpu import VoiceSpline, load_signatures
import numpy as np

sigs, meta = load_signatures('path/to/signatures/')
spline = VoiceSpline(device='cpu')

# Known task positions (e.g., 2D coordinates of task types)
task_pos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Query a new task at position (0.5, 0.5)
query = np.array([0.5, 0.5])

result = spline.fit_predict(task_pos, sigs, query)
print(f"Interpolated signature confidence: {result['confidence']:.2f}")
```

### Model Clustering

```python
from cast_gpu import ModelCluster, load_signatures

sigs, meta = load_signatures('path/to/signatures/')

# Group signatures by generating model
model_sigs = {}
for i, m in enumerate(meta):
    model = m.get('generating_model', 'unknown')
    if model not in model_sigs:
        model_sigs[model] = []
    model_sigs[model].append(sigs[i])
model_sigs = {k: np.array(v) for k, v in model_sigs.items()}

clusterer = ModelCluster(device='auto')
results = clusterer.cluster_by_model('DeepSeek v4-flash', model_sigs, n_clusters=3)
```

---

## Theory

Every piece of text has a **signature** — an N-dimensional vector of structural
features (word length, sentence length, lexical diversity, opening strategy,
negative space use, etc.).

Each model produces a **voice spline** — the manifold of signatures it tends to
produce across different tasks. When a new task type is proposed, we interpolate
along the spline to predict which model will handle it best.

### Signature Distance Matrix

Given N signatures, the distance matrix D is the (N × N) matrix where:
- D[i,j] = cosine distance between signature i and signature j
- D[i,i] = 0
- D is symmetric

Small D[i,j] → similar style/structure between texts i and j
Large D[i,j] → different style/structure

### FLUX Bytecode Connection

Signatures ARE FLUX bytecode. The distance matrix verifies mimicry constraints
directly — if two texts have distance < 0.1 but come from different models,
one model may be mimicking the other's style.

---

## File Structure

```
casting-call-gpu/
├── cast_gpu.py           — GPU-accelerated signature mathematics
├── tests/
│   └── test_cast_gpu.py  — Unit tests
├── examples/
│   └── cluster_demo.py   — Clustering demo
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.9+
- numpy
- scipy (for RBF interpolation)
- Optional: cupy (NVIDIA GPU) or torch (GPU)

---

## License

MIT

---

## Self-Supervised Orchestrator (Next Phase)

The GPU engine evolves from analyzer to autonomous model router:

```
Task with requested signature → find best model → load → run → measure → log → improve
```

### Model Pool Discovery
- Scans for local GGUF models in `~/.cache/llama.cpp/` and similar paths
- Reads API keys from environment or `.keys/` directory
- Tags each model: type, path, context window, speed, cost/token

### Dynamic Load/Unload
- Tracks GPU VRAM per model
- Loads best model match for each task
- Unloads LRU model when VRAM is full
- Frequently used models stay hot in VRAM

### Self-Supervised Learning
- Every task produces: (requested_sig, actual_sig, model, quality)
- Models that consistently match signatures get higher priority
- Models that drift from their stored signature get retested
- No human labels needed — quality is signature distance

### Implementation
```python
def route_task(task, requested_signature):
    model = find_best_model(requested_signature)  # GPU signature lookup
    load_model(model)                               # Dynamic GGUF load
    output = model.generate(task)                    # Run task
    actual_sig = signature(output)                   # Measure output
    log(model, requested_sig, actual_sig)            # Self-supervised log
    return output
```
