# quantization-poc

A proof-of-concept for running quantized GGUF models locally with Metal acceleration on Apple Silicon.

## Results

On an M-series MacBook Air I benchmarked Mistral-7B Q4_K_M: ~3 GB RAM and ~25 tok/s,
measured. Compared against the expected FP16 footprint (~14 GB, ~7 tok/s), that's
roughly 4x the throughput and ~78% less memory — though the FP16 side is an estimate,
not a measured baseline.

## Setup

The commands below reproduce the environment from scratch.

### 1. Initialize the project

```bash
uv init quantization-poc
cd quantization-poc
```

### 2. Add psutil

```bash
uv add psutil
```

### 3. Pin Python to 3.12

`llama-cpp-python` prebuilt wheels lag new Python releases, so pin to 3.12.

In `pyproject.toml`:

```toml
requires-python = ">=3.12,<3.13"
```

In `.python-version`:

```
3.12
```

Then re-sync to recreate the venv on 3.12:

```bash
uv sync
```

### 4. Install llama-cpp-python with Metal

The prebuilt Metal wheels fail to unzip under uv (trailing bytes after the ZIP
end-of-central-directory record). Build from source instead, which compiles
llama.cpp with the Metal backend:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv add llama-cpp-python --no-binary llama-cpp-python
```

> Requires Xcode command-line tools (`xcode-select --install`). The native
> compile takes ~12 minutes.

### 5. Verify Metal support

```bash
uv run python -c "from llama_cpp import llama_cpp; print('Metal:', bool(llama_cpp.llama_supports_gpu_offload()))"
```

Expected output ends with `Metal: True`.

## Usage

Load a model and offload all layers to the GPU:

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", n_gpu_layers=-1)
```


```
source .venv/bin/activate

mkdir ~/models

cd ~/models

curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" -o mistral-7b-instruct-v0.2.Q4_K_M.gguf

cd /Users/sandeepkapoor/Desktop/prep/quantization-poc

python benchmark_gguf.py

(quantization-poc) sandeepkapoor@Sandeeps-MacBook-Air quantization-poc % python benchmark_gguf.py
=================================================================
  GGUF QUANTIZATION BENCHMARK — Apple Silicon MacBook Air
=================================================================

  Model file : /Users/sandeepkapoor/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  File size  : 4.07 GB
  GPU layers : ALL (Metal)
  Context    : 2048 tokens

─────────────────────────────────────────────────────────────────
  STEP 1: Loading model into memory...
─────────────────────────────────────────────────────────────────
llama_context: n_ctx_seq (2048) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
  ✅  Model loaded
  Cold-start load time:               3.2 seconds
  RAM used by model:                  3102 MB  (3.03 GB)
  Total process RAM now:              3121 MB  (3.05 GB)

─────────────────────────────────────────────────────────────────
  STEP 2: Warm-up run (Metal compiles shaders on first call)...
─────────────────────────────────────────────────────────────────
  ✅  Warm-up done. Benchmarks below are post-warm-up.

─────────────────────────────────────────────────────────────────
  STEP 3: Running benchmark prompts...
─────────────────────────────────────────────────────────────────

  [1/3] Short prompt (10 words)
    Prompt tokens:                    11
    Generated tokens:                 50
    Total time:                       2.03s
    Throughput:                       24.6 tok/s  ← KEY NUMBER
    TTFT (approx):                    38 ms

  Output preview: "Machine learning is a subset of artificial intelligence that uses algorithms to enable computers to learn and improve from experience without being explicitly programmed. It involves feeding large amo..."

  [2/3] Medium prompt (30 words)
    Prompt tokens:                    34
    Generated tokens:                 100
    Total time:                       4.15s
    Throughput:                       24.1 tok/s  ← KEY NUMBER
    TTFT (approx):                    57 ms

  Output preview: "Supervised learning and unsupervised learning are two fundamental types of machine learning methods used for training models to make predictions or discover hidden patterns from data.

In supervised l..."

  [3/3] Long prompt (80 words)
    Prompt tokens:                    104
    Generated tokens:                 100
    Total time:                       3.93s
    Throughput:                       25.5 tok/s  ← KEY NUMBER
    TTFT (approx):                    41 ms

  Output preview: "(1) INT4 quantization at the weight level refers to the process of reducing the precision of model weights from the floating-point format (typically FP16 or FP32) to 4-bit integers. In other words, ea..."


=================================================================
  BENCHMARK SUMMARY — Numbers to quote in interviews
=================================================================

  Model      : Mistral-7B Q4_K_M (GGUF)
  Hardware   : MacBook Air Apple Silicon, 24 GB unified memory
  GPU layers : ALL offloaded to Metal

  Metric                              Value
  ─────────────────────────────────── ────────────────────
  Model file size on disk:            4.07 GB
  RAM consumed by model:              3.03 GB
  Cold-start load time:               3.2s
  Avg throughput (all prompts):       24.7 tok/s
  Avg TTFT:                           45 ms

  vs. FP16 baseline (estimated for same hardware):
  ─────────────────────────────────── ────────────────────
    FP16 model size:                  ~14.0 GB
    FP16 throughput (est.):           ~7 tok/s
    RAM saved by INT4:                ~11.0 GB  (78% reduction)
    Speed-up vs FP16 (est.):          3.5x faster

=================================================================
  INTERVIEW STORY:
  'I benchmarked Mistral-7B Q4_K_M on my MacBook Air M-series.
   Model loaded in 3s, used 3.0 GB RAM (vs ~14 GB for FP16),
   and generated at 25 tok/s — roughly 4x faster than
   FP16 on the same hardware. Zero cloud cost, runs offline.'
=================================================================
```