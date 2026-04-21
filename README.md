# TurboCAG — Cache-Augmented Generation with Compressed KV

TurboCAG is a research system for **zero-prefill RAG inference** on NVIDIA GPUs.  
Documents are encoded offline, KV caches are compressed with 4-bit CUDA kernels, and stored to disk. At query time, the model loads the pre-compressed KV and skips prefill entirely — delivering **1.6–1.7× TTFT speedup** vs normal RAG on LongBench/qasper (200 samples, 8K ctx) and **3.5–3.8× VRAM reduction** on Qwen2.5-3B.

---

## How It Works

```
Offline (once per corpus)
──────────────────────────────────────────────────────────────────
Document text
    │
    ▼  forward-pass Qwen2.5 → extract KV per layer
    │
    ▼  compress with TurboQuant CUDA kernels
    │   turbo_prod : K=4-bit interleaved (3-bit Lloyd-Max | 1-bit QJL sign), V=4-bit   → 3.8× compression
    │   turbo_mse  : INT4 MSE-optimal                                                  → 3.5× compression
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

### LongBench / gov\_report — 32K context (99 samples, all padded to 32 768 tokens)

Dataset: [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) `gov_report` test split — long government-report summarisation.  
Each sample is padded to exactly **32 768 tokens** by concatenating neighbouring documents.  
TTFT = disk\_load(36 layers) + query\_prefill. Normal-RAG prefill at 32K tokens is ~5–6 s and near-OOM; CAG eliminates it entirely.

| Scheme     | CAG TTFT (avg) | p50      | p95      | KV VRAM    | VRAM×    |
|------------|---------------|----------|----------|------------|----------|
| fp16 CAG   | 1 538.7 ms    | 1 538 ms | 1 550 ms | **1 152 MB** | 1.0×   |
| turbo\_prod | **1 102.8 ms** | 1 110 ms | 1 133 ms | 302 MB    | **3.8×** |
| turbo\_mse  | 1 137.2 ms    | 1 148 ms | 1 168 ms | 324 MB    | **3.6×** |

Key takeaways at 32K context:
- **TTFT gap: 436 ms** (fp16 → turbo_prod), 28% faster decode start
- **VRAM gap: 3.8×** — turbo_prod saves ~850 MB per inference vs fp16 KV
- 1 doc skipped (OOM during 32K forward-pass precompute); 99/100 succeeded

### LongBench / qasper — 200 samples, ctx=8 192 tokens

Dataset: `qasper` test split — single-doc QA on scientific papers.  
TTFT = disk\_load(36 layers) + query\_prefill. F1 = token-level F1 vs ground truth.  
Normal RAG (full doc+query prefill, 3-sample avg): **2 447 ms**.

| Scheme      | CAG TTFT avg | p50      | p95      | Speedup  | KV VRAM  | VRAM×    | F1    |
|-------------|-------------|----------|----------|----------|----------|----------|-------|
| fp16        | 1 523.8 ms  | 1 505.6  | 1 534.7  | **1.6×** | 288 MB   | 1.0×     | 0.135 |
| turbo\_prod  | 1 402.6 ms  | 1 395.5  | 1 435.3  | **1.7×** | 75 MB    | **3.8×** | 0.125 |
| turbo\_mse   | 1 405.7 ms  | 1 396.9  | 1 451.9  | **1.7×** | 81 MB    | **3.6×** | 0.096 |

### TTFT Scaling — Sim Mode (Qwen2.5-3B scale, 1 143 tokens, 36 layers)

| Scheme     | TTFT    | Speedup vs FP16 prefill | VRAM×    |
|------------|---------|-------------------------|----------|
| FP16       | 129.2 ms | 1.0×                   | 1.0×     |
| turbo\_prod | 3.4 ms  | **37.9×**               | 3.6×     |
| turbo\_mse  | 3.5 ms  | **36.9×**               | 3.2×     |

> Sim mode measures pure GPU disk-load latency vs FP16 prefill (no model weights needed).  
> Full-inference speedup is lower because query prefill is shared across all schemes.

---

## Project Layout

```
TurboCAG/
├── CMakeLists.txt            # CUDA library build
├── include/                  # C++ / CUDA headers
│   ├── tq_config.h
│   ├── tq_turbo_prod.cuh     # turbo_prod layout + kernel declarations
│   ├── tq_turbo_mse_layout.h
│   └── ...
├── src/
│   ├── tq_capi.cpp           # extern "C" API surface
│   └── cuda/
│       ├── tq_turbo_prod_kernels.cu
│       └── tq_turbo_mse_kernels.cu
├── build/
│   └── libturboquant.so      # Compiled shared library
├── tq_backend/               # Python CAG pipeline
│   ├── __init__.py
│   ├── turboquant_wrapper.py # ctypes bindings for all three schemes
│   ├── cag_store.py          # Offline pack + online load per layer
│   ├── model_runner.py       # Qwen2.5 loader + KV extraction
│   ├── benchmark.py          # End-to-end TTFT + accuracy comparison
│   └── ttft_sim.py           # GPU-only TTFT simulation (no model needed)
├── tests/
│   └── test_turbo_prod_nibble.cu  # Roundtrip + nibble bit-order + layout correctness (14 assertions)
├── scripts/
│   ├── build_data.py         # Scan docs (txt/md/jsonl/csv/pdf) → corpus.jsonl + optional KV precompute
│   ├── precompute_cag.py     # Offline: compress corpus to disk (low-level)
│   ├── migrate_store.py      # Migrate old MD5-named stores to SHA-256 naming
│   └── run_benchmark.py      # --mode sim | --mode full | --mode longbench
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
  --quant-type fp16,turbo_prod,turbo_mse
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
  --schemes fp16,turbo_prod,turbo_mse
```

### 3a. GPU TTFT simulation (no model download)

```bash
python scripts/run_benchmark.py --mode sim --tokens 1143 --layers 36
```

### 3b. LongBench benchmark (TTFT + VRAM + F1)

```bash
pip install datasets   # one-time

python scripts/run_benchmark.py \
  --mode longbench \
  --dataset qasper \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-samples 20 --max-length 32768

# Also supports: --dataset 2wikimqa,gov_report  --ttft-only
```

### 3c. Full end-to-end benchmark (custom corpus)

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
    schemes=["fp16", "turbo_prod", "turbo_mse"],
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

pool, slots, N = cag.load_document("doc1", layer_idx=0, scheme="turbo_prod",
                                   head_shape=(2, 128))
output = cag.fused_attention(query, pool, slots, N, "turbo_prod", (2, 128))
# output: [1, 2, 128] fp16 — softmax-weighted sum, no FP16 KV materialised
```

---

## Compression Schemes

| Scheme      | K bits | V bits | Method                   | Compression | AttnMSE   |
|-------------|--------|--------|--------------------------|-------------|-----------|
| turbo\_prod  | 4 (interleaved) | 4 | Lloyd-Max + QJL residual sign (nibble: bits[2:0]=code, bit[3]=sign) | **3.8×** | 0.00028 |
| turbo\_mse   | 4      | 4      | INT4 MSE-optimal         | **3.5×**    | 0.00078   |

All kernels use **online softmax** (FlashAttention-style) — no FP16 KV tensor is ever written to global memory during decode.

---

## Design Notes

**CAG vs RAG.**  Standard RAG prefills the full document context on every query (O(L·N) GPU compute). TurboCAG pre-computes KV offline and loads from disk, reducing per-query GPU work to O(L·disk\_IO + query\_tokens).

**Accuracy trade-off.** turbo\_prod achieves F1=0.125 vs fp16 F1=0.135 on qasper (200 samples, 8K ctx) at 3.8× VRAM savings (AttnMSE=0.00028). The K nibble format packs the 3-bit Lloyd-Max code and 1-bit QJL residual sign into a single 4-bit value per element, preserving fidelity while simplifying the page layout.

**Paged allocation.** `TQAllocator` manages a GPU page pool with `block_size=16` token slots. Slot mappings from vLLM/HF pass directly to TQ kernels — no translation needed.

**GPU startup.** Build `libturboquant.so` with `cmake --build build` before running any Python code.

**KV store versioning.** The binary page format is not versioned. After any K/V layout change, delete all affected `.bin` files and re-run `precompute_cag.py` — loading stale pages with a new kernel silently produces garbage output.
