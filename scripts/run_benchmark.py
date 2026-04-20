#!/usr/bin/env python3
"""run_benchmark.py — End-to-end TTFT + accuracy comparison.

Two modes:
  --mode sim   : GPU-only TTFT simulation, no model download needed (fast)
  --mode full  : Real Qwen2.5-0.5B inference with accuracy measurement

Examples:
  python scripts/run_benchmark.py --mode sim --tokens 512 --layers 24
  python scripts/run_benchmark.py --mode full --store ./kv_store
  python scripts/run_benchmark.py --mode full --corpus data/corpus.jsonl --queries data/queries.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    choices=["sim", "full"], default="sim")
    # sim-mode args
    ap.add_argument("--tokens",  type=int, default=512)
    ap.add_argument("--layers",  type=int, default=24)
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--iters",   type=int, default=20)
    # full-mode args
    ap.add_argument("--model",   default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--store",   default="./kv_store")
    ap.add_argument("--corpus",  default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--schemes", default="fp16,turbo_prod,turbo_mse,polar")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--lib",     default=None)
    args = ap.parse_args()

    if args.mode == "sim":
        from tq_backend.ttft_sim import run_ttft_sim
        run_ttft_sim(
            num_tokens=args.tokens,
            num_layers=args.layers,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        from tq_backend.benchmark import run_benchmark
        run_benchmark(
            model_name=args.model,
            store_dir=args.store,
            corpus_path=args.corpus,
            queries_path=args.queries,
            schemes=args.schemes.split(","),
            max_new_tokens=args.new_tokens,
            lib_path=args.lib,
        )


if __name__ == "__main__":
    main()
