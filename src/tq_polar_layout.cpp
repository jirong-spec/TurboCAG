#include "tq_polar_layout.h"
#include <cuda_fp16.h>

static inline size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQPolarPageLayout make_tq_polar_page_layout(const TQConfig& cfg) {
    TQPolarPageLayout l;

    l.k4_bytes_per_token_head  = cfg.head_dim / 2;              // 4 bits/dim
    l.v4_bytes_per_token_head  = cfg.head_dim / 2;              // 4 bits/dim
    l.scale_bytes_per_token_head = static_cast<int>(sizeof(half));

    const size_t token_heads =
        static_cast<size_t>(cfg.block_size) * cfg.num_kv_heads;

    const size_t k4_region = token_heads * l.k4_bytes_per_token_head;
    const size_t ks_region = token_heads * l.scale_bytes_per_token_head;
    const size_t v4_region = token_heads * l.v4_bytes_per_token_head;
    const size_t vs_region = token_heads * l.scale_bytes_per_token_head;

    size_t cursor = 0;
    l.k4_codes_offset = cursor; cursor += k4_region;
    cursor = align_up(cursor, alignof(half));
    l.k_scales_offset = cursor; cursor += ks_region;
    l.v4_codes_offset = cursor; cursor += v4_region;
    cursor = align_up(cursor, alignof(half));
    l.v_scales_offset = cursor; cursor += vs_region;
    l.page_size_bytes = align_up(cursor, alignof(half));

    return l;
}
