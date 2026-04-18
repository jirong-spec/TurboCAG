#!/usr/bin/env python3
# Claude's version — config-driven RAG benchmark with TurboQuant KV simulation
"""rag_turbo_demo.py

Bring-your-own-data RAG benchmark powered by TurboQuant.
All dataset / model / output settings live in ../config.yaml.

Pipeline
────────
1. Load corpus from KaggleHub or a local CSV (config.yaml → dataset)
2. Build Q&A pairs or load your own (config.yaml → qa)
3. BM25 retrieval over all questions → retrieval recall
4. Ollama LLM inference on a sample   → LLM accuracy
5. TurboQuant KV simulation on every retrieved context
6. Write output/rag_results.md  +  output/rag_results.json

Usage
─────
    python3 rag_turbo_demo.py                   # uses ../config.yaml
    python3 rag_turbo_demo.py --config my.yaml  # custom config path
    python3 rag_turbo_demo.py --llm-sample 100  # override one field

Bring your own data
───────────────────
Edit config.yaml:
  1. Set dataset.source to "local" and dataset.local_path to your CSV, or
     set dataset.kaggle_handle / dataset.kaggle_file for a Kaggle dataset.
  2. Set dataset.text_columns to the column(s) that form the RAG corpus.
  3. Set qa.auto_generate: false and supply your own JSONL at qa.path, or
     configure qa.templates to match your dataset's structured columns.
  4. Adjust llm.model, llm.sample_size, retrieval.top_k as needed.
  5. Run: python3 rag_turbo_demo.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# Optional Kaggle support
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    _KAGGLE_OK = True
except ImportError:
    _KAGGLE_OK = False

# Optional dense retrieval + cross-encoder re-ranking
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _ST_OK = True
except ImportError:
    _ST_OK = False

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_wrapper import TurboQuantWrapper

DEVICE = "cuda"
CHARS_PER_TOK = 3.5


# ─────────────────────────────────────────────────────────────────── #
# Config loading                                                       #
# ─────────────────────────────────────────────────────────────────── #

def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _cfg(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ─────────────────────────────────────────────────────────────────── #
# Dataset loading                                                      #
# ─────────────────────────────────────────────────────────────────── #

def load_dataframe(cfg: dict) -> "pd.DataFrame":
    source = _cfg(cfg, "dataset", "source", default="kaggle")
    nrows  = _cfg(cfg, "dataset", "nrows",  default=5000)

    if source == "local":
        local_path = _cfg(cfg, "dataset", "local_path")
        if not local_path:
            sys.exit("config.yaml: dataset.local_path must be set when source=local")
        p = Path(local_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"Local dataset not found: {p}")
        print(f"Loading   : {p} (nrows={nrows})")
        return pd.read_csv(p, nrows=nrows)

    # Kaggle
    if not _KAGGLE_OK:
        sys.exit("pip install kagglehub  (required for source=kaggle)")
    handle = _cfg(cfg, "dataset", "kaggle_handle")
    file   = _cfg(cfg, "dataset", "kaggle_file")
    if not handle:
        sys.exit("config.yaml: dataset.kaggle_handle must be set")
    print(f"Fetching  : {handle} / {file} …")
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        handle,
        file or "",
        pandas_kwargs={"nrows": nrows},
    )


def build_corpus(df: "pd.DataFrame", text_cols: list[str]) -> list[str]:
    available = [c for c in text_cols if c in df.columns]
    if not available:
        sys.exit(f"None of text_columns {text_cols!r} found. Available: {df.columns.tolist()}")
    combined = df[available[0]].fillna("").astype(str)
    for col in available[1:]:
        combined = combined + ". " + df[col].fillna("").astype(str)
    return combined.str.strip().tolist()


# ─────────────────────────────────────────────────────────────────── #
# QA generation / loading                                              #
# ─────────────────────────────────────────────────────────────────── #

def generate_qa_pairs(
    df: "pd.DataFrame",
    name_col: str,
    templates: dict[str, str],
    n: int = 5_000,
) -> list[dict]:
    pairs: list[dict] = []
    for _, row in df.iterrows():
        name = row.get(name_col, "")
        if not name or pd.isna(name):
            continue
        for field, tmpl in templates.items():
            val = row.get(field)
            if pd.notna(val) and str(val).strip() not in ("nan", ""):
                pairs.append({
                    "question":    tmpl.format(name=str(name).strip()),
                    "answer":      str(val).strip(),
                    "source_name": str(name).strip(),
                    "field":       field,
                })
        if len(pairs) >= n:
            break
    return pairs[:n]


def load_or_generate_qa(df: "pd.DataFrame", cfg: dict) -> list[dict]:
    qa_path = ROOT / _cfg(cfg, "qa", "path", default="data/qa_pairs.jsonl")
    auto    = _cfg(cfg, "qa", "auto_generate", default=True)

    if qa_path.exists():
        print(f"QA file   : {qa_path} (cached)")
        with qa_path.open() as f:
            return [json.loads(l) for l in f if l.strip()]

    if not auto:
        sys.exit(
            f"qa.auto_generate is false but {qa_path} does not exist.\n"
            "Either set auto_generate: true or provide the JSONL file."
        )

    name_col   = _cfg(cfg, "qa", "name_column", default="name")
    templates  = _cfg(cfg, "qa", "templates", default={})
    nrows      = _cfg(cfg, "dataset", "nrows", default=5000)

    if not templates:
        sys.exit(
            "config.yaml: qa.templates is empty. "
            "Define at least one field→template mapping."
        )
    if name_col not in df.columns:
        sys.exit(
            f"config.yaml: qa.name_column='{name_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    print(f"Generating QA pairs → {qa_path} …")
    pairs = generate_qa_pairs(df, name_col, templates, n=nrows)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"QA file   : saved {len(pairs)} pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────── #
# GPU-BM25: index stored as sparse CUDA tensor in VRAM               #
# ─────────────────────────────────────────────────────────────────── #

def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(re.findall(r'[\d,.]+\s*[%]?|[A-Za-z]+', text))
    cjk = re.sub(r'[^\u4e00-\u9fff]', '', text)
    tokens.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))
    return tokens


class BM25:
    """BM25 index stored as a sparse CSR tensor in VRAM.

    Build (CPU): tokenise → compute BM25 TF weights → upload sparse matrix.
    Query (GPU): vectorise query → sparse matrix-vector multiply → top-k.

    Memory: ~8 bytes × non-zeros.  For 200k docs × 50 unique terms ≈ 80 MB VRAM.
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75,
                 device: str = DEVICE) -> None:
        self.docs   = docs
        self.N      = len(docs)
        self.device = device

        # ── Build vocabulary and raw TF on CPU ──────────────────────── #
        vocab: dict[str, int] = {}
        tokenized: list[list[str]] = []
        for doc in docs:
            toks = _tokenize(doc)
            tokenized.append(toks)
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        V = len(vocab)

        doc_rows, term_cols, tf_vals = [], [], []
        df_cpu = [0] * V
        dl = []

        for d_idx, toks in enumerate(tokenized):
            c = Counter(toks)
            dl.append(len(toks))
            for term, freq in c.items():
                t_idx = vocab[term]
                doc_rows.append(d_idx)
                term_cols.append(t_idx)
                tf_vals.append(float(freq))
                df_cpu[t_idx] += 1

        avgdl = sum(dl) / max(self.N, 1)
        dl_t  = torch.tensor(dl, dtype=torch.float32)

        # ── IDF ─────────────────────────────────────────────────────── #
        df_t  = torch.tensor(df_cpu, dtype=torch.float32)
        idf_t = torch.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1.0)
        self.idf = idf_t.to(device)   # [V]  — stays in VRAM

        # ── BM25 TF weights (CPU, then upload) ──────────────────────── #
        rows_t  = torch.tensor(doc_rows,  dtype=torch.long)
        cols_t  = torch.tensor(term_cols, dtype=torch.long)
        tf_t    = torch.tensor(tf_vals,   dtype=torch.float32)
        dl_rows = dl_t[rows_t]

        bm25_tf = tf_t * (k1 + 1.0) / (
            tf_t + k1 * (1.0 - b + b * dl_rows / avgdl)
        )

        # Sparse COO → CSR, upload to VRAM
        # Shape: [N_docs × V];  scores = matrix @ (q_vec ⊙ idf)
        sparse_coo = torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t]),
            bm25_tf,
            (self.N, V),
            dtype=torch.float32,
        ).coalesce()
        self.matrix = sparse_coo.to_sparse_csr().to(device)  # VRAM

        vram_mb = (bm25_tf.numel() * 4 + rows_t.numel() * 8) / 1024**2
        print(f"GPU-BM25  : {self.N} docs, vocab {V:,}, "
              f"nnz {bm25_tf.numel():,} (~{vram_mb:.0f} MB VRAM)")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float, str]]:
        qtoks = _tokenize(query)
        if not qtoks:
            return []

        # Query vector on GPU
        q_vec = torch.zeros(len(self.vocab), dtype=torch.float32, device=self.device)
        for t in qtoks:
            idx = self.vocab.get(t)
            if idx is not None:
                q_vec[idx] += 1.0

        # Scores = sparse_matrix @ (q_vec * idf)   shape: [N_docs]
        scores = self.matrix @ (q_vec * self.idf)

        k = min(k, self.N)
        topk = torch.topk(scores, k)
        return [
            (idx.item(), topk.values[i].item(), self.docs[idx.item()])
            for i, idx in enumerate(topk.indices)
            if topk.values[i].item() > 0
        ]


# ─────────────────────────────────────────────────────────────────── #
# Dense retriever: embeddings in VRAM, cosine similarity              #
# ─────────────────────────────────────────────────────────────────── #

class DenseRetriever:
    """Encode corpus with sentence-transformers; store float16 embeddings in VRAM.

    Memory: N_docs × D × 2 bytes.  all-MiniLM-L6-v2 (D=384):
      200k docs → 154 MB VRAM.
    Query latency: single GPU matmul, sub-millisecond.
    """

    def __init__(
        self,
        docs: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = DEVICE,
        batch_size: int = 512,
    ) -> None:
        if not _ST_OK:
            raise ImportError("pip install sentence-transformers")
        self.docs   = docs
        self.device = device

        print(f"Dense     : loading {model_name} …")
        self._model = SentenceTransformer(model_name, device=device)

        print(f"Dense     : encoding {len(docs):,} docs (batch={batch_size}) …")
        emb = self._model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,   # L2-norm → cosine sim = dot product
        )
        self.embeddings = emb.half()     # [N, D] float16 in VRAM
        vram_mb = self.embeddings.numel() * 2 / 1024 ** 2
        print(f"Dense     : dim={emb.shape[1]}, {vram_mb:.0f} MB VRAM")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float, str]]:
        q = self._model.encode(
            [query], convert_to_tensor=True, device=self.device,
            normalize_embeddings=True,
        ).half()                                       # [1, D]
        scores = (self.embeddings @ q.T).squeeze(1)   # [N]
        k      = min(k, len(self.docs))
        topk   = torch.topk(scores, k)
        return [
            (idx.item(), topk.values[i].item(), self.docs[idx.item()])
            for i, idx in enumerate(topk.indices)
        ]


# ─────────────────────────────────────────────────────────────────── #
# Hybrid retriever: RRF(BM25, Dense)                                  #
# ─────────────────────────────────────────────────────────────────── #

class HybridRetriever:
    """Reciprocal Rank Fusion of BM25 and dense results.

    RRF score = Σ 1/(rrf_k + rank_i),  rrf_k=60 (standard default).
    Fetches k*3 candidates from each arm before fusion so tail docs
    still get a chance to surface in the merged top-k.
    """

    def __init__(self, bm25: BM25, dense: DenseRetriever, rrf_k: int = 60) -> None:
        self.bm25   = bm25
        self.dense  = dense
        self.rrf_k  = rrf_k

    @property
    def docs(self) -> list[str]:
        return self.bm25.docs

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float, str]]:
        fetch = k * 3
        bm25_hits  = self.bm25.retrieve(query,  k=fetch)
        dense_hits = self.dense.retrieve(query, k=fetch)

        rrf: dict[int, float] = {}
        for rank, (doc_idx, _, _) in enumerate(bm25_hits):
            rrf[doc_idx] = rrf.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (doc_idx, _, _) in enumerate(dense_hits):
            rrf[doc_idx] = rrf.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(doc_idx, score, self.bm25.docs[doc_idx]) for doc_idx, score in top]


# ─────────────────────────────────────────────────────────────────── #
# Cross-encoder re-ranker                                             #
# ─────────────────────────────────────────────────────────────────── #

class CrossEncoderReranker:
    """Score (query, doc) pairs with a cross-encoder; return top-k by score.

    Model default: cross-encoder/ms-marco-MiniLM-L-6-v2 (~65 MB).
    Runs on GPU; typical latency for 20 pairs < 20 ms.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = DEVICE,
    ) -> None:
        if not _ST_OK:
            raise ImportError("pip install sentence-transformers")
        print(f"Reranker  : loading {model_name} …")
        self._model = CrossEncoder(model_name, device=device)
        print(f"Reranker  : ready")

    def rerank(
        self,
        query: str,
        hits: list[tuple[int, float, str]],
        k: int = 5,
    ) -> list[tuple[int, float, str]]:
        if not hits:
            return hits
        pairs  = [(query, doc) for _, _, doc in hits]
        scores = self._model.predict(pairs, batch_size=32)  # batch to avoid OOM
        ranked = sorted(zip(scores.tolist(), hits), key=lambda x: x[0], reverse=True)
        return [hit for _, hit in ranked[:k]]


# ─────────────────────────────────────────────────────────────────── #
# Ollama                                                               #
# ─────────────────────────────────────────────────────────────────── #

def ollama_available(url: str, model: str) -> bool:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:
        return False


def ollama_generate(
    prompt: str, url: str, model: str, timeout: int
) -> tuple[str, float, float]:
    """Returns (text, total_ms, ttft_ms). Uses streaming to capture TTFT."""
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    parts: list[str] = []
    try:
        with requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=timeout,
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                tok = data.get("response", "")
                if tok and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                parts.append(tok)
                if data.get("done"):
                    break
        total_ms = (time.perf_counter() - t0) * 1000
        return "".join(parts).strip(), total_ms, ttft_ms or 0.0
    except Exception as exc:
        return f"[unavailable: {exc}]", 0.0, 0.0


def build_prompt(system: str, question: str, context: str) -> str:
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


# ─────────────────────────────────────────────────────────────────── #
# TurboQuant KV simulation                                            #
# ─────────────────────────────────────────────────────────────────── #

@dataclass
class KVResult:
    scheme: str; num_tokens: int
    fp16_mb: float; quant_mb: float; compression: float
    pack_us: float; kv_mse: float


def _bench(fn, warmup=3, iters=20) -> float:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def simulate_kv(tq: TurboQuantWrapper, num_tokens: int) -> list[KVResult]:
    cfg = tq.default_config()
    H, D = cfg.num_kv_heads, cfg.head_dim
    k = torch.randn(num_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    v = torch.randn(num_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    s = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    fp16_b = tq.fp16_bytes(num_tokens, cfg)
    results: list[KVResult] = []

    lay = tq.make_layout_for(cfg)
    pool = tq.alloc_page_pool(num_tokens, lay, cfg)
    ok = torch.empty_like(k); ov = torch.empty_like(v)
    pu = _bench(lambda: tq.pack(k, v, s, pool, lay, cfg))
    tq.pack(k, v, s, pool, lay, cfg); torch.cuda.synchronize()
    tq.dequant(pool, s, ok, ov, lay, cfg); torch.cuda.synchronize()
    qb = tq.quant_bytes(num_tokens, lay, cfg)
    results.append(KVResult("turbo_prod", num_tokens, fp16_b/1024**2, qb/1024**2,
                             fp16_b/qb, pu,
                             TurboQuantWrapper.compute_mse(torch.cat([ok,ov]), torch.cat([k,v]))))
    try:
        ml = tq.make_mse_layout_for(cfg)
        mp = tq.alloc_mse_pool(num_tokens, ml, cfg)
        mk = torch.empty_like(k); mv = torch.empty_like(v)
        mpu = _bench(lambda: tq.mse_pack(k, v, s, mp, ml, cfg))
        tq.mse_pack(k, v, s, mp, ml, cfg); torch.cuda.synchronize()
        tq.mse_dequant(mp, s, mk, mv, ml, cfg); torch.cuda.synchronize()
        mb = tq.mse_bytes(num_tokens, ml, cfg)
        results.append(KVResult("turbo_mse", num_tokens, fp16_b/1024**2, mb/1024**2,
                                 fp16_b/mb, mpu,
                                 TurboQuantWrapper.compute_mse(torch.cat([mk,mv]), torch.cat([k,v]))))
    except Exception as exc:
        warnings.warn(f"turbo_mse skipped: {exc}", stacklevel=2)
    return results


# ─────────────────────────────────────────────────────────────────── #
# Report                                                               #
# ─────────────────────────────────────────────────────────────────── #

def _avg(lst: list[KVResult], attr: str) -> float:
    return sum(getattr(r, attr) for r in lst) / len(lst) if lst else float("nan")


def write_report(
    cfg: dict,
    qa_pairs: list[dict],
    bm25_hits: int,
    hybrid_hits: int | None,
    reranked_hits: int | None,
    llm_rows: list[dict],
    kv_all: list[KVResult],
) -> None:
    out_dir   = ROOT / _cfg(cfg, "output", "dir",         default="output")
    stem      = _cfg(cfg, "output", "report_stem", default="rag_results")
    top_k     = _cfg(cfg, "retrieval", "top_k",    default=5)
    fetch_k   = _cfg(cfg, "retrieval", "fetch_k",  default=top_k)
    model     = _cfg(cfg, "llm", "model",          default="")
    handle    = _cfg(cfg, "dataset", "kaggle_handle", default="local")
    out_dir.mkdir(parents=True, exist_ok=True)

    total          = len(qa_pairs)
    bm25_recall    = bm25_hits    / total if total else 0
    hybrid_recall  = hybrid_hits  / total if (hybrid_hits  is not None and total) else None
    reranked_recall= reranked_hits / total if (reranked_hits is not None and total) else None
    llm_acc        = (sum(1 for r in llm_rows if r["correct"]) / len(llm_rows)
                      if llm_rows else None)

    prod_lst  = [r for r in kv_all if r.scheme == "turbo_prod"]
    mse_lst   = [r for r in kv_all if r.scheme == "turbo_mse"]

    if reranked_hits is not None:
        retrieval_mode = f"Hybrid@{fetch_k} + CrossEncoder → top-{top_k}"
    elif hybrid_hits is not None:
        retrieval_mode = f"Hybrid (BM25 + Dense, RRF), top-{top_k}"
    else:
        retrieval_mode = f"BM25, top-{top_k}"

    lines = [
        "# TurboRAG — RAG Benchmark Report\n",
        f"- Dataset      : `{handle}`",
        f"- Corpus size  : {total} documents",
        f"- Questions    : {total}",
        f"- LLM model    : {model}",
        f"- Retrieval    : {retrieval_mode}",
        "",
        "---",
        "",
        "## Retrieval Recall\n",
        f"| Retriever | k | Hits | Recall |",
        f"|-----------|---|------|--------|",
        f"| BM25 only | {top_k} | {bm25_hits} | {bm25_recall:.1%} |",
    ]
    if hybrid_recall is not None:
        delta = hybrid_recall - bm25_recall
        lines.append(
            f"| Hybrid (RRF) | {fetch_k} | {hybrid_hits} | {hybrid_recall:.1%} "
            f"(+{delta:.1%}) |"
        )
    if reranked_recall is not None:
        delta2 = reranked_recall - bm25_recall
        lines.append(
            f"| **Hybrid + Rerank** | **{top_k}** | **{reranked_hits}** | **{reranked_recall:.1%}** "
            f"(+{delta2:.1%} vs BM25) |"
        )
    lines += ["", "---", ""]

    if llm_rows:
        correct_n = sum(1 for r in llm_rows if r["correct"])
        avg_lat   = sum(r["lat_ms"]  for r in llm_rows) / len(llm_rows)
        avg_ttft  = sum(r.get("ttft_ms", 0) for r in llm_rows) / len(llm_rows)
        lines += [
            f"## LLM Accuracy ({model}, {len(llm_rows)}-question sample)\n",
            "| # | Question | Expected | Correct | TTFT ms | Total ms |",
            "|---|----------|----------|---------|---------|----------|",
        ]
        for r in llm_rows:
            tick = "✓" if r["correct"] else "✗"
            lines.append(
                f"| {r['qi']} | {r['question'][:55]} | {r['expected'][:25]} "
                f"| {tick} | {r.get('ttft_ms',0):.0f} | {r['lat_ms']:.0f} |"
            )
        lines += [
            "",
            f"**LLM accuracy: {correct_n}/{len(llm_rows)} = {llm_acc:.1%}**  "
            f"avg TTFT {avg_ttft:.0f} ms · avg total {avg_lat:.0f} ms",
            "",
            "---",
            "",
        ]

    lines += [
        "## KV-Cache Efficiency (TurboQuant, averaged over all questions)\n",
        "| Scheme | Avg Tokens | FP16 MB | Quant MB | Compression | Pack µs | KV MSE |",
        "|--------|-----------|---------|---------|-------------|---------|--------|",
    ]
    for scheme, lst in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]:
        if not lst:
            continue
        lines.append(
            f"| {scheme} | {_avg(lst,'num_tokens'):.0f} "
            f"| {_avg(lst,'fp16_mb'):.3f} "
            f"| {_avg(lst,'quant_mb'):.3f} "
            f"| {_avg(lst,'compression'):.2f}× "
            f"| {_avg(lst,'pack_us'):.1f} "
            f"| {_avg(lst,'kv_mse'):.3e} |"
        )
    lines.append("| FP16 baseline | — | — | — | 1.00× | — | 0 |")
    lines += ["", "---", "", "## Summary\n"]
    lines.append(f"- BM25 recall            : **{bm25_recall:.1%}** ({bm25_hits}/{total})")
    if hybrid_recall is not None:
        lines.append(
            f"- Hybrid recall @{fetch_k}     : **{hybrid_recall:.1%}** ({hybrid_hits}/{total})"
            f"  (+{hybrid_recall-bm25_recall:.1%})"
        )
    if reranked_recall is not None:
        lines.append(
            f"- Reranked recall @{top_k}    : **{reranked_recall:.1%}** ({reranked_hits}/{total})"
            f"  (+{reranked_recall-bm25_recall:.1%} vs BM25)"
        )
    if llm_acc is not None:
        lines.append(
            f"- LLM answer accuracy    : **{llm_acc:.1%}** ({correct_n}/{len(llm_rows)} sample)"
        )
        lines.append(
            f"- LLM avg TTFT           : **{avg_ttft:.0f} ms**  (total {avg_lat:.0f} ms)"
        )
    for scheme, lst in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]:
        c = _avg(lst, "compression")
        if not math.isnan(c):
            lines.append(f"- {scheme} compression  : **{c:.2f}×** VRAM vs FP16")

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "bm25_recall":     bm25_recall,
        "hybrid_recall":   hybrid_recall,
        "reranked_recall": reranked_recall,
        "llm_accuracy":    llm_acc,
        "llm_rows":       llm_rows,
        "kv_summary": {
            s: {"avg_compression": _avg(l,"compression"),
                "avg_pack_us":     _avg(l,"pack_us"),
                "avg_kv_mse":      _avg(l,"kv_mse")}
            for s, l in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nReport    : {md_path}")
    print(f"JSON      : {json_path}")
    print("\n" + "─" * 60)
    print("\n".join(lines[-10:]))


# ─────────────────────────────────────────────────────────────────── #
# Main                                                                 #
# ─────────────────────────────────────────────────────────────────── #

def run(cfg: dict, llm_sample_override: int | None) -> None:
    # ── Load corpus ─────────────────────────────────────────────── #
    df        = load_dataframe(cfg)
    text_cols = _cfg(cfg, "dataset", "text_columns", default=["description"])
    corpus    = build_corpus(df, text_cols)
    print(f"Corpus    : {len(df)} rows, {len(corpus)} documents")

    # ── QA pairs ────────────────────────────────────────────────── #
    qa_pairs  = load_or_generate_qa(df, cfg)
    llm_sample = llm_sample_override or _cfg(cfg, "llm", "sample_size", default=50)
    print(f"Questions : {len(qa_pairs)} total  (LLM sample: {llm_sample})\n")

    # ── BM25 ────────────────────────────────────────────────────── #
    print("Building BM25 index …")
    bm25_index   = BM25(corpus)
    top_k        = _cfg(cfg, "retrieval", "top_k",        default=5)
    fetch_k      = _cfg(cfg, "retrieval", "fetch_k",      default=20)
    ret_mode     = _cfg(cfg, "retrieval", "mode",         default="hybrid")
    dense_model  = _cfg(cfg, "retrieval", "dense_model",  default="sentence-transformers/all-MiniLM-L6-v2")
    rrf_k        = _cfg(cfg, "retrieval", "rrf_k",        default=60)
    reranker_mdl = _cfg(cfg, "retrieval", "reranker",     default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Dense + Hybrid ──────────────────────────────────────────── #
    retriever: BM25 | HybridRetriever = bm25_index
    use_hybrid = (ret_mode == "hybrid")
    if use_hybrid:
        if not _ST_OK:
            print("Dense     : sentence-transformers not installed → falling back to BM25")
            use_hybrid = False
        else:
            try:
                dense     = DenseRetriever(corpus, model_name=dense_model)
                retriever = HybridRetriever(bm25_index, dense, rrf_k=rrf_k)
                print(f"Retriever : Hybrid (BM25 + Dense, RRF k={rrf_k}), fetch_k={fetch_k}")
            except Exception as exc:
                print(f"Dense     : failed ({exc}) → falling back to BM25")
                use_hybrid = False

    if not use_hybrid:
        print("Retriever : BM25 only")

    # ── Cross-encoder re-ranker ─────────────────────────────────── #
    reranker: CrossEncoderReranker | None = None
    if use_hybrid and reranker_mdl:
        if not _ST_OK:
            print("Reranker  : sentence-transformers not installed → skipped")
        else:
            try:
                reranker = CrossEncoderReranker(model_name=reranker_mdl)
            except Exception as exc:
                print(f"Reranker  : failed ({exc}) → skipped")

    # ── GPU + TurboQuant ────────────────────────────────────────── #
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required for TurboQuant simulation.")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    tq = TurboQuantWrapper()

    # ── Ollama ──────────────────────────────────────────────────── #
    ollama_url = _cfg(cfg, "llm", "ollama_url",  default="http://localhost:11434")
    model      = _cfg(cfg, "llm", "model",       default="qwen2.5:7b")
    timeout    = _cfg(cfg, "llm", "timeout_sec", default=120)
    system     = _cfg(cfg, "llm", "system_prompt",
                      default="You are a helpful assistant. Answer using ONLY the context.")
    ollama_ok  = ollama_available(ollama_url, model)
    print(f"Ollama    : {'✓ ' + model if ollama_ok else '✗ not available'}\n")

    # ── Main loop ───────────────────────────────────────────────── #
    print(f"Running retrieval + KV simulation on {len(qa_pairs)} questions …")
    bm25_hit_count    = 0
    hybrid_hit_count  = 0 if use_hybrid else None
    reranked_hit_count= 0 if reranker   else None
    kv_all:   list[KVResult] = []
    llm_rows: list[dict]     = []

    for qi, qa in enumerate(qa_pairs):
        ans_low = qa["answer"].lower()

        # BM25@top_k baseline (always tracked)
        bm25_hits_q = bm25_index.retrieve(qa["question"], k=top_k)
        if ans_low in " ".join(d.lower() for _, _, d in bm25_hits_q):
            bm25_hit_count += 1

        # Hybrid@fetch_k → optional Reranked@top_k
        if use_hybrid:
            hybrid_hits_q = retriever.retrieve(qa["question"], k=fetch_k)
            if ans_low in " ".join(d.lower() for _, _, d in hybrid_hits_q):
                hybrid_hit_count += 1

            if reranker:
                reranked_q = reranker.rerank(qa["question"], hybrid_hits_q, k=top_k)
                if ans_low in " ".join(d.lower() for _, _, d in reranked_q):
                    reranked_hit_count += 1
                hits_for_context = reranked_q
            else:
                hits_for_context = hybrid_hits_q[:top_k]
        else:
            hits_for_context = bm25_hits_q

        context  = "\n\n".join(doc for _, _, doc in hits_for_context)
        n_tokens = min(max(64, int(len(context) / CHARS_PER_TOK)), 4096)
        kv_all.extend(simulate_kv(tq, n_tokens))

        if ollama_ok and qi < llm_sample:
            answer, lat_ms, ttft_ms = ollama_generate(
                build_prompt(system, qa["question"], context),
                ollama_url, model, timeout,
            )
            correct = ans_low in answer.lower()
            llm_rows.append({
                "qi": qi + 1, "question": qa["question"],
                "expected": qa["answer"],
                "answer": answer[:200].replace("\n", " "),
                "correct": correct, "lat_ms": lat_ms, "ttft_ms": ttft_ms,
            })
            if (qi + 1) % 10 == 0:
                print(f"  LLM {qi+1}/{llm_sample} done …")

        if (qi + 1) % 500 == 0:
            pct  = bm25_hit_count / (qi + 1) * 100
            hyb  = f"  hybrid@{fetch_k}={hybrid_hit_count/(qi+1)*100:.1f}%" if use_hybrid else ""
            rnk  = f"  rerank@{top_k}={reranked_hit_count/(qi+1)*100:.1f}%" if reranker   else ""
            print(f"  [{qi+1:>5}/{len(qa_pairs)}] BM25@{top_k}={pct:.1f}%{hyb}{rnk}")

    write_report(cfg, qa_pairs, bm25_hit_count, hybrid_hit_count,
                 reranked_hit_count, llm_rows, kv_all)


def main() -> None:
    p = argparse.ArgumentParser(description="TurboRAG benchmark — edit config.yaml to use your own data")
    p.add_argument("--config",     default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    p.add_argument("--llm-sample", type=int, default=None, help="Override llm.sample_size")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    run(cfg, args.llm_sample)


if __name__ == "__main__":
    main()
