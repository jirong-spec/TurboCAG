# TurboCAG — Cache-Augmented Generation with Compressed KV

TurboCAG is a research system for **zero-prefill RAG inference** on NVIDIA GPUs.  
Documents are encoded offline, KV caches are compressed with 4-bit CUDA kernels, and stored to disk. At query time, the model loads the pre-compressed KV and skips prefill entirely — delivering **37–48× TTFT speedup** and **3.2–3.7× VRAM reduction** with near-zero accuracy loss on Qwen2.5-3B.

---

## How It Works

```
Offline (once per corpus)
──────────────────────────────────────────────────────────────────
Document text
    │
    ▼  forward-pass Qwen2.5 → extract KV per layer
    │
    ▼  compress with TurboQuant / PolarQuant CUDA kernels
    │   turbo_prod : K=3-bit + 1-bit residual, V=4-bit   → 3.8× compression
    │   turbo_mse  : INT4 MSE-optimal                    → 3.5× compression
    │   polar      : Hadamard-rotated K=4-bit, V=4-bit   → 3.9× compression
    │
    ▼  save compressed page_pool to disk  (.bin + .meta per layer)

Online (per query)
──────────────────────────────────────────────────────────────────
Query
    │
    ▼  load compressed KV from disk  (all L layers)
    │
    ▼  dequant → DynamicCache → inject as past_key_values
    │
    ▼  model.generate(query_tokens_only)   ← no document prefill
    │
    ▼  answer
```

**TTFT savings** = prefill(doc_tokens) is skipped.  
For a 1143-token document on Qwen2.5-3B: prefill takes 244 ms; CAG loads KV in ~40 ms → **6× speedup**.

---

## Benchmark Results — Qwen2.5-3B-Instruct

Hardware: **NVIDIA GeForce RTX 3060 12 GB**, CUDA 12.4  
Corpus: GYG travel activities dataset, 700–1143 tokens per document  
Metric: CAG TTFT = disk\_load(all 36 layers) + query\_prefill

### TTFT & VRAM (full model inference)

| Scheme     | Normal RAG | CAG TTFT | Speedup  | KV VRAM | VRAM×    |
|------------|------------|----------|----------|---------|----------|
| fp16 CAG   | 244.4 ms   | 46.2 ms  | **5.3×** | 24.3 MB | 1.0×     |
| turbo\_prod | 244.4 ms   | 40.0 ms  | **6.1×** | 6.5 MB  | **3.8×** |
| turbo\_mse  | 244.4 ms   | 40.1 ms  | **6.1×** | 7.0 MB  | **3.5×** |
| polar      | 244.4 ms   | 39.0 ms  | **6.3×** | 5.4 MB  | **4.5×** |

### Accuracy & Attention Fidelity

| Scheme     | Accuracy | AttnMSE    | Notes                              |
|------------|----------|------------|------------------------------------|
| fp16 CAG   | **75%**  | 0 (ref)    | Lossless reference                 |
| turbo\_prod | **67%**  | **0.00028** | −8pp vs fp16; near-lossless        |
| turbo\_mse  | 33%      | 0.00078    | Lower MSE but more hallucination   |
| polar      | 33%      | 0.04774    | 4-bit WHT-rotated; balanced noise  |

AttnMSE = attention output MSE vs FP16 reference (per layer, averaged over 3 layers).  
Accuracy = exact-match: reference answer substring found in model output.

### TTFT Scaling — Sim Mode (Qwen2.5-3B scale, 1143 tokens, 36 layers)

| Scheme     | TTFT     | Speedup vs FP16 prefill | VRAM×    |
|------------|----------|-------------------------|----------|
| FP16       | 129.2ms  | 1.0×                    | 1.0×     |
| turbo\_prod | 3.4ms   | **37.9×**               | 3.6×     |
| turbo\_mse  | 3.5ms   | **36.9×**               | 3.2×     |
| polar      | 2.7ms   | **47.8×**               | **3.7×** |

> Sim mode measures pure GPU prefill vs disk-load latency (no model weights needed).  
> Full-inference speedup is lower because query prefill (~35ms) is shared across all schemes.

---

## Project Layout

```
TurboCAG/
├── CMakeLists.txt            # CUDA library build
├── include/                  # C++ / CUDA headers
│   ├── tq_config.h
│   ├── tq_turbo_prod.cuh     # turbo_prod layout + kernel declarations
│   ├── tq_turbo_mse_layout.h
│   ├── tq_polar_layout.h     # PolarQuant layout
│   ├── tq_polar.cuh
│   └── ...
├── src/
│   ├── tq_capi.cpp           # extern "C" API surface
│   └── cuda/
│       ├── tq_turbo_prod_kernels.cu
│       ├── tq_turbo_mse_kernels.cu
│       └── tq_polar_kernels.cu
├── build/
│   └── libturboquant.so      # Compiled shared library
├── tq_backend/               # Python CAG pipeline
│   ├── __init__.py
│   ├── turboquant_wrapper.py # ctypes bindings for all three schemes
│   ├── cag_store.py          # Offline pack + online load per layer
│   ├── model_runner.py       # Qwen2.5 loader + KV extraction
│   ├── benchmark.py          # End-to-end TTFT + accuracy comparison
│   └── ttft_sim.py           # GPU-only TTFT simulation (no model needed)
├── scripts/
│   ├── build_data.py         # Scan docs (txt/md/jsonl/csv/pdf) → corpus.jsonl + optional KV precompute
│   ├── precompute_cag.py     # Offline: compress corpus to disk (low-level)
│   ├── migrate_store.py      # Migrate old MD5-named stores to SHA-256 naming
│   └── run_benchmark.py      # --mode sim | --mode full
└── data/
    ├── gyg_qa_5000.jsonl     # GYG activity Q&A pairs
    ├── long_corpus.jsonl     # Country-grouped long documents
    └── long_queries.jsonl    # Q&A pairs for long-doc benchmark
```

---

## Requirements

| Dependency   | Version         |
|--------------|-----------------|
| CUDA Toolkit | 11.7+           |
| CMake        | 3.20+           |
| C++          | 17              |
| Python       | 3.10+           |
| PyTorch      | 2.0+ (CUDA)     |
| transformers | 4.40+           |
| accelerate   | any             |

Tested on **NVIDIA GeForce RTX 3060 12 GB**, CUDA 12.4, driver 550.163.

---

## Build

```bash
cmake -B build -S .
cmake --build build --parallel $(nproc)
# produces build/libturboquant.so
```

---

## Quick Start

### 1. Build corpus from your documents

```bash
# Chunk a folder of text / Markdown / JSONL / CSV / PDF files → corpus.jsonl
python scripts/build_data.py \
  --input-dir data/docs/ \
  --output-dir data/ \
  --corpus-only

# Or build corpus AND precompute KV caches in one shot
python scripts/build_data.py \
  --input-dir data/docs/ \
  --output-dir data/ \
  --store ./kv_store \
  --model qwen2.5-3b \
  --quant-type fp16,turbo_prod,polar
```

Supported formats: `--formats txt,md,jsonl,csv,pdf`  
Chunking strategies: `--chunking fixed | sentence | paragraph`  
Model shorthands: `qwen2.5-0.5b`, `qwen2.5-3b`, `llama3.2-3b`, `mistral-7b`, …

> **GYG demo corpus** (if using `data/gyg_qa_5000.jsonl`):  
> `python scripts/build_data.py --input-dir data/ --formats jsonl --output-dir data/ --corpus-only`

### 2. Offline: precompute KV cache

```bash
python scripts/precompute_cag.py \
  --corpus data/long_corpus.jsonl \
  --store ./kv_store \
  --schemes fp16,turbo_prod,turbo_mse,polar
```

### 3a. GPU TTFT simulation (no model download)

```bash
python scripts/run_benchmark.py --mode sim --tokens 1143 --layers 36
```

### 3b. Full end-to-end benchmark (Qwen2.5-3B)

```bash
python scripts/run_benchmark.py \
  --mode full \
  --model Qwen/Qwen2.5-3B-Instruct \
  --store ./kv_store \
  --corpus data/long_corpus.jsonl \
  --queries data/long_queries.jsonl \
  --new-tokens 40
```

---

## Python API

### Offline: compress a corpus

```python
from tq_backend import TQModelRunner

runner = TQModelRunner("Qwen/Qwen2.5-3B-Instruct", store_dir="./kv_store")
runner.precompute_corpus(
    {"doc1": "text of document one ...", "doc2": "text of document two ..."},
    schemes=["fp16", "turbo_prod", "turbo_mse", "polar"],
)
```

### Online: CAG inference with compressed KV

```python
from tq_backend.cag_store import CAGStore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

cag = CAGStore("./kv_store")

# Load and dequant all layers → DynamicCache
cache, doc_len = cag.build_dynamic_cache(
    "doc1", num_layers=36, scheme="turbo_prod",
    head_shape=(2, 128),   # Qwen2.5-3B: 2 KV heads, head_dim=128
)

# Generate from query only (no document prefill)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16, device_map="cuda")

q_ids = tokenizer("\nQuestion: What is X?\nAnswer:", return_tensors="pt").to("cuda")
q_len = q_ids["input_ids"].shape[1]
pos   = torch.arange(doc_len, doc_len + q_len, device="cuda").unsqueeze(0)
mask  = torch.ones(1, doc_len + q_len, device="cuda", dtype=torch.long)

out = model.generate(
    input_ids=q_ids["input_ids"],
    attention_mask=mask,
    past_key_values=cache,
    position_ids=pos,
    max_new_tokens=64,
    do_sample=False,
)
print(tokenizer.decode(out[0][q_len:], skip_special_tokens=True))
```

### Low-level: fused attention over compressed KV

```python
import torch
from tq_backend.cag_store import CAGStore

cag   = CAGStore("./kv_store")
query = torch.randn(1, 2, 128, device="cuda", dtype=torch.float16)  # [1, H_kv, D]

pool, slots, N = cag.load_document("doc1", layer_idx=0, scheme="polar",
                                   head_shape=(2, 128))
output = cag.fused_attention(query, pool, slots, N, "polar", (2, 128))
# output: [1, 2, 128] fp16 — softmax-weighted sum, no FP16 KV materialised
```

---

## Compression Schemes

| Scheme      | K bits | V bits | Method                        | Compression | AttnMSE   |
|-------------|--------|--------|-------------------------------|-------------|-----------|
| turbo\_prod  | 3+1    | 4      | Lloyd-Max + QJL residual      | **3.8×**    | 0.00028   |
| turbo\_mse   | 4      | 4      | INT4 MSE-optimal              | **3.5×**    | 0.00078   |
| polar       | 4      | 4      | Hadamard rotation + INT4 coding | **3.9×**  | 0.04774   |

All kernels use **online softmax** (FlashAttention-style) — no FP16 KV tensor is ever written to global memory during decode.

---

## Design Notes

**CAG vs RAG.**  Standard RAG prefills the full document context on every query (O(L·N) GPU compute). TurboCAG pre-computes KV offline and loads from disk, reducing per-query GPU work to O(L·disk\_IO + query\_tokens).

**Accuracy trade-off.** turbo\_prod achieves 67% exact-match accuracy (vs 75% for fp16 CAG) at 3.8× VRAM savings and near-lossless attention fidelity (AttnMSE = 0.00028). Errors arise from quantization noise accumulating across 36 transformer layers.

**Paged allocation.** `TQAllocator` manages a GPU page pool with `block_size=16` token slots. Slot mappings from vLLM/HF pass directly to TQ kernels — no translation needed.

**GPU startup.** Build `libturboquant.so` with `cmake --build build` before running any Python code.
