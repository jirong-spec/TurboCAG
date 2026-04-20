#pragma once
// Host-side CUDA error-check helpers shared by all TurboQuant kernel files.
//
// TQ_CHECK_LAUNCH(name)
//   Call immediately after <<<...>>>.  Catches invalid launch configuration
//   (bad grid/block dims, shmem too large) via cudaGetLastError().
//
// TQ_CHECK_ASYNC(stream)
//   Catches async kernel execution errors (OOB, illegal addr, etc.) by
//   inserting cudaStreamSynchronize().  Active only when compiled with
//   -DTQ_DEBUG_SYNC; expands to a no-op in release builds.
//   Never enable in hot paths — synchronise adds GPU stall overhead.

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define TQ_CHECK_LAUNCH(kernel_name)                                           \
    do {                                                                       \
        cudaError_t _tq_le = cudaGetLastError();                               \
        if (_tq_le != cudaSuccess)                                             \
            throw std::runtime_error(                                          \
                std::string(kernel_name ": ") + cudaGetErrorString(_tq_le));   \
    } while (0)

#ifdef TQ_DEBUG_SYNC
  #define TQ_CHECK_ASYNC(stream)                                               \
    do {                                                                       \
        cudaError_t _tq_ae = cudaStreamSynchronize(stream);                    \
        if (_tq_ae != cudaSuccess)                                             \
            throw std::runtime_error(                                          \
                std::string("async kernel error: ") +                         \
                cudaGetErrorString(_tq_ae));                                   \
    } while (0)
#else
  #define TQ_CHECK_ASYNC(stream) ((void)(stream))
#endif
