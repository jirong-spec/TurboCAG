#include "tq_polar.cuh"
#include "tq_shared_device.cuh"
#include "tq_cuda_check.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half  f2h(float x) { return __float2half(x); }

template<typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float sign_flip(int idx) {
    unsigned x = static_cast<unsigned>(idx) * 1103515245u + 12345u;
    return (x & 1u) ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// Codebook — 4-bit, 16-level Gaussian Lloyd-Max optimal for N(0,1)
// Shared by both K and V in the WHT-rotated domain.
// ---------------------------------------------------------------------------

__device__ __constant__ float kPolarCodebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

__device__ __forceinline__ int nearest_polar4_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kPolarCodebook[0]);
    #pragma unroll
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(x - kPolarCodebook[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// In-place normalised Walsh-Hadamard Transform (butterfly on shared memory).
// Requires all D threads alive; leaves them synchronised at exit.
// Replicated per translation unit (same as V6 / QJL).
// ---------------------------------------------------------------------------
template<int MAX_D>
__device__ void hadamard_inplace(float* x, int D) {
    for (int len = 1; len < D; len <<= 1) {
        int tid      = threadIdx.x;
        int butterfly = D >> 1;
        if (tid < butterfly) {
            int group  = tid / len;
            int offset = tid % len;
            int i0 = group * (len << 1) + offset;
            int i1 = i0 + len;
            float a = x[i0], b = x[i1];
            x[i0] = a + b;
            x[i1] = a - b;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Pack kernel
//
// grid = (num_tokens, num_kv_heads),  threads = head_dim
// Shared memory (extern): sk[MAX_D] | sv[MAX_D] | red[MAX_D]
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_pack_kv_kernel(
    const half* __restrict__   key,
    const half* __restrict__   value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__      page_pool,
    TQPolarPageLayout          layout,
    TQConfig                   cfg,
    int                        num_tokens)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot           = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    uint8_t* k4_codes = page_base + layout.k4_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.k4_bytes_per_token_head);
    half* kscale = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.v4_bytes_per_token_head);
    half* vscale = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk  = smem;           // [MAX_D] rotated K
    float* sv  = smem + MAX_D;   // [MAX_D] rotated V
    float* red = smem + 2*MAX_D; // [MAX_D] reduction scratch

    __shared__ int kidx_s[MAX_D]; // 4-bit K indices
    __shared__ int vidx_s[MAX_D]; // 4-bit V indices

    const int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    const float inv_sqrt_d = rsqrtf((float)D);

    // ---- Step 1: sign-flip + WHT + normalise K and V ----------------------
    sk[tid] = h2f(key  [base + tid]) * sign_flip(tid);
    sv[tid] = h2f(value[base + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    sk[tid] *= inv_sqrt_d;
    sv[tid] *= inv_sqrt_d;

    // ---- Step 2: K RMS scale ----------------------------------------------
    red[tid] = sk[tid] * sk[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float krms = sqrtf(red[0] / (float)D);
    if (krms < kMinRMS) krms = kMinRMS;
    if (tid == 0) *kscale = f2h(krms);
    __syncthreads();

    // ---- Step 3: V RMS scale ----------------------------------------------
    red[tid] = sv[tid] * sv[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float vrms = sqrtf(red[0] / (float)D);
    if (vrms < kMinRMS) vrms = kMinRMS;
    if (tid == 0) *vscale = f2h(vrms);
    __syncthreads();

    // ---- Step 4: quantise (4-bit, 16-level Gaussian codebook) ------------
    kidx_s[tid] = nearest_polar4_idx(sk[tid] / krms);
    vidx_s[tid] = nearest_polar4_idx(sv[tid] / vrms);
    __syncthreads();

    // ---- Step 5: pack K (4-bit nibbles, 2 codes/byte) --------------------
    if (tid < D / 2)
        k4_codes[tid] = (uint8_t)((kidx_s[2*tid+1] << 4) | (kidx_s[2*tid] & 0xF));

    // ---- Step 6: pack V (4-bit nibbles, 2 codes/byte) --------------------
    if (tid < D / 2)
        v4_codes[tid] = (uint8_t)((vidx_s[2*tid+1] << 4) | (vidx_s[2*tid] & 0xF));
}

// ---------------------------------------------------------------------------
// Dequant kernel
//
// grid = (num_tokens, num_kv_heads),  threads = head_dim
// Shared memory (extern): smem[MAX_D]  (reused for K then V)
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_dequant_kv_kernel(
    const uint8_t* __restrict__  page_pool,
    const int32_t* __restrict__  slot_mapping,
    half* __restrict__           out_key,
    half* __restrict__           out_value,
    TQPolarPageLayout            layout,
    TQConfig                     cfg,
    int                          num_tokens)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot           = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    const uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    const uint8_t* k4_codes = page_base + layout.k4_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.k4_bytes_per_token_head);
    const half* k_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.v4_bytes_per_token_head);
    const half* v_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    extern __shared__ float smem[]; // [MAX_D]

    const int base        = (token_idx * cfg.num_kv_heads + head_idx) * D;
    const float inv_sqrt_d = rsqrtf((float)D);

    // ---- Decode K: 4-bit → WHT⁻¹ → sign-unflip → FP16 -------------------
    float krms = h2f(*k_scale_ptr);
    smem[tid]  = kPolarCodebook[unpack_4bit_get(k4_codes, tid)] * krms;
    __syncthreads();

    hadamard_inplace<MAX_D>(smem, D);

    out_key[base + tid] = f2h(smem[tid] * inv_sqrt_d * sign_flip(tid));
    __syncthreads();

    // ---- Decode V: 4-bit → WHT⁻¹ → sign-unflip → FP16 -------------------
    float vrms = h2f(*v_scale_ptr);
    smem[tid]  = kPolarCodebook[unpack_4bit_get(v4_codes, tid)] * vrms;
    __syncthreads();

    hadamard_inplace<MAX_D>(smem, D);

    out_value[base + tid] = f2h(smem[tid] * inv_sqrt_d * sign_flip(tid));
}

// ---------------------------------------------------------------------------
// Fused attention: online softmax → weighted-V sum → inverse rotation.
//
// grid = (num_queries, num_kv_heads),  threads = head_dim
// Shared memory (extern): qrot[MAX_D] | vaccum[MAX_D] | red[MAX_D]
//
// K decoded as 4-bit, V decoded as 4-bit (both in Hadamard-rotated domain).
// Logit convention: <q, k>  (no 1/sqrt(D)), matching V6 / QJL convention.
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_fused_attn_online_kernel(
    const half* __restrict__    query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__          output,
    TQPolarPageLayout           layout,
    TQConfig                    cfg,
    int                         num_queries,
    int                         num_kv_tokens)
{
    int q_idx    = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    const float inv_sqrt_d = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* qrot   = smem;           // [MAX_D] rotated query
    float* vaccum = smem + MAX_D;   // [MAX_D] weighted V accumulation
    float* red    = smem + 2*MAX_D; // [MAX_D] reduction scratch

    __shared__ float sh_m, sh_l, sh_a, sh_b;

    // ---- Rotate query: sign_flip + WHT + 1/sqrt(D) ------------------------
    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(qrot, D);
    qrot[tid] *= inv_sqrt_d;
    __syncthreads();

    vaccum[tid] = 0.0f;
    if (tid == 0) { sh_m = kInitMaxLogit; sh_l = 0.0f; }
    __syncthreads();

    for (int t = 0; t < num_kv_tokens; ++t) {
        int slot           = slot_mapping[t];
        int physical_block = slot / cfg.block_size;
        int token_in_block = slot % cfg.block_size;

        const uint8_t* page_base = page_pool +
            (size_t)physical_block * layout.page_size_bytes;

        // ---- 4-bit K decode → dot product ----------------------------------
        const uint8_t* k4_codes = page_base + layout.k4_codes_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.k4_bytes_per_token_head);
        const half* k_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.scale_bytes_per_token_head));

        float kval = kPolarCodebook[unpack_4bit_get(k4_codes, tid)] * h2f(*k_scale_ptr);

        red[tid] = qrot[tid] * kval;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // red[0] = logit = <qrot, krot>

        // ---- Online softmax update (thread 0) ------------------------------
        if (tid == 0) {
            float logit = red[0];
            float m_new = fmaxf(sh_m, logit);
            sh_a = expf(sh_m - m_new);
            sh_b = expf(logit - m_new);
            sh_l = sh_l * sh_a + sh_b;
            sh_m = m_new;
        }
        __syncthreads();

        vaccum[tid] *= sh_a;

        // ---- 4-bit V decode → accumulate -----------------------------------
        const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.v4_bytes_per_token_head);
        const half* v_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.v_scales_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.scale_bytes_per_token_head));

        float vval = kPolarCodebook[unpack_4bit_get(v4_codes, tid)] * h2f(*v_scale_ptr);

        vaccum[tid] += sh_b * vval;
        __syncthreads();
    }

    // ---- Normalise + inverse WHT + sign-unflip → output -------------------
    __shared__ float sh_inv_l;
    if (tid == 0) sh_inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 1.0f;
    __syncthreads();
    vaccum[tid] *= sh_inv_l;

    __syncthreads();
    hadamard_inplace<MAX_D>(vaccum, D);

    int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = f2h(vaccum[tid] * inv_sqrt_d * sign_flip(tid));
}

} // namespace

// ---------------------------------------------------------------------------
// Launch functions
// ---------------------------------------------------------------------------

void launch_tq_polar_pack_kv(
    const half*              key,
    const half*              value,
    const int32_t*           slot_mapping,
    uint8_t*                 page_pool,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar pack: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    // smem: sk[128] + sv[128] + red[128]  (static __shared__ is separate)
    size_t shmem = sizeof(float) * 3 * 128;
    polar_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
    TQ_CHECK_LAUNCH("polar_pack_kv_kernel");
    TQ_CHECK_ASYNC(stream);
}

void launch_tq_polar_dequant_kv(
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    out_key,
    half*                    out_value,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar dequant: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * 128;
    polar_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
    TQ_CHECK_LAUNCH("polar_dequant_kv_kernel");
    TQ_CHECK_ASYNC(stream);
}

void launch_tq_polar_fused_attention_output(
    const half*              query,
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    output,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_queries,
    int                      num_kv_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar fused attn: head_dim > 128 not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    // smem: qrot[128] + vaccum[128] + red[128]  (sh_* are static __shared__)
    size_t shmem = sizeof(float) * 3 * 128;
    polar_fused_attn_online_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, output,
        layout, cfg, num_queries, num_kv_tokens);
    TQ_CHECK_LAUNCH("polar_fused_attn_online_kernel");
    TQ_CHECK_ASYNC(stream);
}
