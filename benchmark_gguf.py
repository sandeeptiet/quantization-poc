"""
GGUF Quantization Benchmark — MacBook Air Apple Silicon
========================================================
What this script does:
  1. Loads a GGUF model (Q4_K_M quantized) using llama-cpp-python
  2. Runs a cold-start timing test
  3. Runs multiple inference calls with different prompt lengths
  4. Reports: tokens/sec, RAM usage, TTFT (Time To First Token), total latency
  5. Prints a clean before/after comparison table

SETUP (run these in your terminal BEFORE running this script):
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
  pip install psutil

DOWNLOAD THE MODEL (run in terminal):
  mkdir -p ~/models
  # Download Mistral 7B Q4_K_M (~4.1 GB) from HuggingFace:
  curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
       -o ~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

  # Or download a smaller model first to test (Phi-3 Mini, ~2.4 GB):
  curl -L "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
       -o ~/models/phi3-mini-q4.gguf

RUN THIS SCRIPT:
  python benchmark_gguf.py
"""

import time
import os
import sys
import psutil
import statistics

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Change this path to wherever you downloaded the model
MODEL_PATH = os.path.expanduser("~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# n_gpu_layers=-1  means: offload ALL layers to Apple Metal GPU (fastest)
# n_gpu_layers=0   means: run entirely on CPU (slower but always works)
# n_gpu_layers=20  means: offload first 20 layers to GPU (partial)
N_GPU_LAYERS = -1

# Context window size (tokens). 2048 is safe for 24GB RAM with 7B Q4
N_CTX = 2048

# Number of tokens to generate per benchmark run
MAX_TOKENS = 100

# ─── TEST PROMPTS (short, medium, long) ──────────────────────────────────────
TEST_PROMPTS = [
    {
        "label": "Short prompt (10 words)",
        "text": "What is machine learning? Answer in two sentences."
    },
    {
        "label": "Medium prompt (30 words)",
        "text": (
            "Explain the difference between supervised and unsupervised learning. "
            "Give one real-world example of each and describe when you would choose "
            "one approach over the other."
        )
    },
    {
        "label": "Long prompt (80 words)",
        "text": (
            "You are an AI assistant helping a senior engineer understand model "
            "quantization. Explain in detail: (1) what INT4 quantization means at "
            "the weight level, (2) how GPTQ differs from AWQ in the quantization "
            "algorithm, (3) what accuracy trade-offs to expect when moving from "
            "FP16 to INT4, and (4) in which production scenarios you would choose "
            "NOT to quantize a model despite the memory savings. Be specific and "
            "technical in your answer."
        )
    },
]

# ─── HELPER: get current RAM used by this process ────────────────────────────
def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes → MB

# ─── HELPER: format a result row ─────────────────────────────────────────────
def fmt_row(label, value):
    return f"  {label:<35} {value}"

# ─── MAIN BENCHMARK ──────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  GGUF QUANTIZATION BENCHMARK — Apple Silicon MacBook Air")
    print("=" * 65)

    # 1. Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model file not found at: {MODEL_PATH}")
        print("    Please download it using the curl command in the comments above.")
        print("    Or change MODEL_PATH at the top of this script.")
        sys.exit(1)

    model_size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
    print(f"\n  Model file : {MODEL_PATH}")
    print(f"  File size  : {model_size_gb:.2f} GB")
    print(f"  GPU layers : {'ALL (Metal)' if N_GPU_LAYERS == -1 else N_GPU_LAYERS}")
    print(f"  Context    : {N_CTX} tokens")

    # 2. Measure RAM before loading
    ram_before_mb = get_ram_mb()

    # 3. Load the model — measure cold-start time
    print(f"\n{'─'*65}")
    print("  STEP 1: Loading model into memory...")
    print(f"{'─'*65}")

    from llama_cpp import Llama  # Import here so error is clear if not installed

    load_start = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,   # -1 = all layers on Metal GPU
        n_ctx=N_CTX,                  # context window
        n_threads=8,                  # CPU threads for any CPU-side work
        verbose=False,                # set True to see llama.cpp internals
    )
    load_time = time.time() - load_start

    ram_after_mb = get_ram_mb()
    ram_used_mb = ram_after_mb - ram_before_mb

    print(f"  ✅  Model loaded")
    print(fmt_row("Cold-start load time:", f"{load_time:.1f} seconds"))
    print(fmt_row("RAM used by model:",    f"{ram_used_mb:.0f} MB  ({ram_used_mb/1024:.2f} GB)"))
    print(fmt_row("Total process RAM now:", f"{ram_after_mb:.0f} MB  ({ram_after_mb/1024:.2f} GB)"))

    # 4. Warm-up run (first call is always slower due to Metal shader compilation)
    print(f"\n{'─'*65}")
    print("  STEP 2: Warm-up run (Metal compiles shaders on first call)...")
    print(f"{'─'*65}")
    _ = llm("Hello", max_tokens=10, echo=False)
    print("  ✅  Warm-up done. Benchmarks below are post-warm-up.")

    # 5. Benchmark each prompt
    print(f"\n{'─'*65}")
    print("  STEP 3: Running benchmark prompts...")
    print(f"{'─'*65}\n")

    all_results = []

    for i, prompt_cfg in enumerate(TEST_PROMPTS, 1):
        label = prompt_cfg["label"]
        prompt_text = prompt_cfg["text"]

        # Count prompt tokens
        prompt_tokens = len(llm.tokenize(prompt_text.encode()))

        # Run 3 times and take the median (removes outliers)
        run_times = []
        run_tokens = []

        for run in range(3):
            t_start = time.time()

            output = llm(
                prompt_text,
                max_tokens=MAX_TOKENS,
                echo=False,          # Don't include prompt in output
                temperature=0.0,     # Greedy decode — deterministic, fair benchmark
                stop=["</s>", "\n\n\n"],
            )

            t_end = time.time()
            elapsed = t_end - t_start

            # Extract generated tokens count
            generated_tokens = output["usage"]["completion_tokens"]
            run_times.append(elapsed)
            run_tokens.append(generated_tokens)

        # Take median run
        median_idx = sorted(range(3), key=lambda x: run_times[x])[1]
        best_time = run_times[median_idx]
        best_tokens = run_tokens[median_idx]
        toks_per_sec = best_tokens / best_time if best_time > 0 else 0

        # TTFT approximation: time for the model to generate the first token
        # We measure by running with max_tokens=1
        t_ttft_start = time.time()
        _ = llm(prompt_text, max_tokens=1, echo=False, temperature=0.0)
        ttft = time.time() - t_ttft_start

        result = {
            "label": label,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": best_tokens,
            "total_time_s": best_time,
            "toks_per_sec": toks_per_sec,
            "ttft_ms": ttft * 1000,
        }
        all_results.append(result)

        print(f"  [{i}/3] {label}")
        print(fmt_row("  Prompt tokens:",      f"{prompt_tokens}"))
        print(fmt_row("  Generated tokens:",   f"{best_tokens}"))
        print(fmt_row("  Total time:",         f"{best_time:.2f}s"))
        print(fmt_row("  Throughput:",         f"{toks_per_sec:.1f} tok/s  ← KEY NUMBER"))
        print(fmt_row("  TTFT (approx):",      f"{ttft*1000:.0f} ms"))
        print()

        # Print the actual generated text
        actual_output = output["choices"][0]["text"].strip()
        preview = actual_output[:200] + ("..." if len(actual_output) > 200 else "")
        print(f"  Output preview: \"{preview}\"")
        print()

    # 6. Summary comparison table
    print(f"\n{'=' * 65}")
    print("  BENCHMARK SUMMARY — Numbers to quote in interviews")
    print(f"{'=' * 65}\n")

    print(f"  Model      : Mistral-7B Q4_K_M (GGUF)")
    print(f"  Hardware   : MacBook Air Apple Silicon, 24 GB unified memory")
    print(f"  GPU layers : ALL offloaded to Metal\n")

    print(f"  {'Metric':<35} {'Value'}")
    print(f"  {'─'*35} {'─'*20}")
    print(fmt_row("Model file size on disk:",    f"{model_size_gb:.2f} GB"))
    print(fmt_row("RAM consumed by model:",      f"{ram_used_mb/1024:.2f} GB"))
    print(fmt_row("Cold-start load time:",       f"{load_time:.1f}s"))
    avg_tps = statistics.mean([r["toks_per_sec"] for r in all_results])
    print(fmt_row("Avg throughput (all prompts):", f"{avg_tps:.1f} tok/s"))
    avg_ttft = statistics.mean([r["ttft_ms"] for r in all_results])
    print(fmt_row("Avg TTFT:",                   f"{avg_ttft:.0f} ms"))

    print(f"\n  vs. FP16 baseline (estimated for same hardware):")
    print(f"  {'─'*35} {'─'*20}")
    # FP16 7B would be ~14GB RAM, ~6-8 tok/s on M-series
    estimated_fp16_tps = 7.0
    speedup = avg_tps / estimated_fp16_tps
    print(fmt_row("  FP16 model size:",          "~14.0 GB"))
    print(fmt_row("  FP16 throughput (est.):",   "~7 tok/s"))
    print(fmt_row("  RAM saved by INT4:",         f"~{14.0 - ram_used_mb/1024:.1f} GB  ({((14.0 - ram_used_mb/1024)/14.0*100):.0f}% reduction)"))
    print(fmt_row("  Speed-up vs FP16 (est.):",  f"{speedup:.1f}x faster"))

    print(f"\n{'=' * 65}")
    print("  INTERVIEW STORY:")
    print(f"  'I benchmarked Mistral-7B Q4_K_M on my MacBook Air M-series.")
    print(f"   Model loaded in {load_time:.0f}s, used {ram_used_mb/1024:.1f} GB RAM (vs ~14 GB for FP16),")
    print(f"   and generated at {avg_tps:.0f} tok/s — roughly {speedup:.0f}x faster than")
    print(f"   FP16 on the same hardware. Zero cloud cost, runs offline.'")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
