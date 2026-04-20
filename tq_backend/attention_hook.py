"""attention_hook.py — Hook-based injection of TQ compressed KV into Qwen2.5.

Two modes:
  profile_mode  : intercept attention forward, measure pack+attn timing
  cag_mode      : skip prefill entirely; load pre-computed KV from disk
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class AttentionStats:
    layer_idx: int
    scheme: str
    pack_us: float = 0.0
    attn_us: float = 0.0
    kv_mse: float = 0.0
    attn_mse: float = 0.0
    num_tokens: int = 0


@dataclass
class HookState:
    scheme: str = "fp16"
    cag_mode: bool = False
    doc_id: str = ""
    stats: list[AttentionStats] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)


def install_tq_hooks(
    model,
    store,           # CAGStore
    scheme: str,
    doc_id: str = "",
    cag_mode: bool = False,
) -> HookState:
    """Register forward pre-hooks on every Qwen2.5 attention layer.

    In profile_mode: pack K,V → run TQ fused attention → compare to FP16 reference.
    In cag_mode:     load pre-packed KV from disk; skip normal attention computation.
    """
    state = HookState(scheme=scheme, cag_mode=cag_mode, doc_id=doc_id)

    cfg = model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.hidden_size // cfg.num_attention_heads

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        head_shape = (num_kv_heads, head_dim)

        def make_hook(lidx, hs):
            def _hook(module, args, kwargs):
                # Only intercept when there's input to process
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.shape[1] == 0:
                    return

                if not hs.cag_mode:
                    return  # profile mode handled in post-hook

                # CAG mode: if doc is pre-computed, replace attention
                if not store.exists(hs.doc_id, lidx, hs.scheme):
                    return

                # Mark that we're handling this attention
                module._tq_cag_active = True

            return _hook

        def make_post_hook(lidx, hs, h_shape):
            def _post_hook(module, args, kwargs, output):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    return output

                stats = AttentionStats(layer_idx=lidx, scheme=hs.scheme)

                if hs.cag_mode and store.exists(hs.doc_id, lidx, hs.scheme):
                    # Load pre-computed KV and run fused attention
                    t0 = time.perf_counter()
                    pool, slots, N = store.load_document(hs.doc_id, lidx, hs.scheme, h_shape)
                    torch.cuda.synchronize()
                    stats.pack_us = (time.perf_counter() - t0) * 1e6
                    stats.num_tokens = N

                    # Build query from hidden states via module's q_proj
                    bsz, seq_len, _ = hidden.shape
                    q = module.q_proj(hidden)
                    H_full = module.num_heads
                    D = module.head_dim
                    H_kv = module.num_key_value_heads
                    q = q.view(bsz, seq_len, H_full, D)

                    # Use first KV head's query for the fused attention
                    groups = H_full // H_kv
                    q_kv = q[:, :, ::groups, :].contiguous()  # [B, S, H_kv, D]
                    q_kv = q_kv.view(bsz * seq_len, H_kv, D)

                    t1 = time.perf_counter()
                    attn_out = store.fused_attention(q_kv[:1], pool, slots, N, hs.scheme, h_shape)
                    torch.cuda.synchronize()
                    stats.attn_us = (time.perf_counter() - t1) * 1e6

                    hs.stats.append(stats)

                return output

            return _post_hook

        pre_h  = attn.register_forward_pre_hook(make_hook(layer_idx, state), with_kwargs=True)
        post_h = attn.register_forward_hook(make_post_hook(layer_idx, state, head_shape), with_kwargs=True)
        state.handles.extend([pre_h, post_h])

    return state


def remove_hooks(state: HookState) -> None:
    for h in state.handles:
        h.remove()
    state.handles.clear()
