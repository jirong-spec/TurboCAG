"""tq_backend — TurboQuant / PolarQuant attention backend for Qwen2.5.

Drop-in replacement for Qwen2.5 attention that uses compressed KV caches.
Supports three compression schemes:
  fp16       — reference, no compression
  turbo_prod — K=3b+1b residual / V=4b  (~15-16× vs FP16 per page)
  turbo_mse  — INT4 MSE-optimised         (~8× vs FP16 per page)
  polar      — K=2-bit / V=3-bit Hadamard  (~6.1× vs FP16 per page)

CAG (Cache-Augmented Generation):
  Offline: precompute_corpus() → per-layer compressed KV stored to disk
  Online:  model.generate() skips prefill; loads KV from disk per layer
"""
from tq_backend.model_runner import TQModelRunner
from tq_backend.cag_store import CAGStore

__all__ = ["TQModelRunner", "CAGStore"]
