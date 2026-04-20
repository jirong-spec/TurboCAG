#pragma once
// ---------------------------------------------------------------------------
// PolarQuant page layout: K=4-bit + V=4-bit in Hadamard-rotated domain
//
// Memory per token-head at head_dim=128:
//   K: 64 B (4-bit codes) + 2 B (FP16 RMS scale) = 66 B
//   V: 64 B (4-bit codes) + 2 B (FP16 RMS scale) = 66 B
//   Total: 132 B  vs  512 B dense  → ~3.9× compression
//
// Uses the same 16-level Gaussian Lloyd-Max codebook as turbo_mse,
// applied in the WHT-rotated domain for both K and V.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include "tq_config.h"

struct TQPolarPageLayout {
    size_t page_size_bytes = 0;

    size_t k4_codes_offset = 0;  // 4-bit K codes  (D/2 bytes per token-head)
    size_t k_scales_offset = 0;  // FP16 K RMS scales
    size_t v4_codes_offset = 0;  // 4-bit V codes  (D/2 bytes per token-head)
    size_t v_scales_offset = 0;  // FP16 V RMS scales

    int k4_bytes_per_token_head = 0;   // = head_dim / 2
    int v4_bytes_per_token_head = 0;   // = head_dim / 2
    int scale_bytes_per_token_head = 0; // = sizeof(half) = 2
};

TQPolarPageLayout make_tq_polar_page_layout(const TQConfig& cfg);

// ---------------------------------------------------------------------------
// Byte offset of a (token_in_block, head_idx) cell inside one region.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define TQ_POLAR_HD __host__ __device__
#else
#define TQ_POLAR_HD
#endif

inline TQ_POLAR_HD size_t polar_token_head_offset(
    int token_in_block,
    int head_idx,
    int num_kv_heads,
    int bytes_per_token_head)
{
    return ((size_t)token_in_block * num_kv_heads + head_idx) * bytes_per_token_head;
}

#undef TQ_POLAR_HD
