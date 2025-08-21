/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_CUH
#define UCT_CUDA_IPC_CUH

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

extern "C" {
#include "cuda_ipc_ep_dev.h"
}

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))
#define WARP_SIZE 32
#define COPY_LOOP_UNROLL 8

__device__ static inline int
uct_cuda_ipc_batch_has_atomic(const uct_cuda_ipc_batch_t *batch)
{
    return batch->list[batch->num - 1].e_op == UCT_CUDA_IPC_ATOMIC_FA;
}

__device__ static inline int
uct_cuda_ipc_batch_has_iov(const uct_cuda_ipc_batch_t *batch)
{
    return batch->list[0].e_op != UCT_CUDA_IPC_ATOMIC_FA;
}

#if ENABLE_PARAMS_CHECK
__device__ static inline ucs_status_t
uct_cuda_ipc_batch_params_check(const uct_cuda_ipc_batch_t *batch, const uint64_t flags,
                                const int has_iov, const int has_atomic,
                                uct_dev_completion_t *comp)
{
    if ((flags & UCT_DEV_BATCH_FLAG_ATOMIC) && !has_atomic) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((flags & UCT_DEV_BATCH_FLAG_RMA_IOV) && !has_iov) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((flags & UCT_DEV_BATCH_FLAG_COMP) && (comp == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}
#endif

/* unused, keep for debugging */
__device__ static inline void uct_cuda_ipc_batch_copy_single_nv(void *dst,
                                                                const void *src,
                                                                size_t size)
{
    size_t i;
    char *s1 = (char *)src;
    char *d1 = (char *)dst;

    for (i = threadIdx.x; i <size; i += blockDim.x) {
        d1[i] = s1[i];
    }
}

static __device__ __forceinline__ int4 ld_global_cg(const int4* p) {
    int4 v;
    asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cg(int4* p, const int4& v) {
    asm volatile ("st.global.cg.v4.s32 [%0], {%1,%2,%3,%4};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

template<int UNROLL>
__device__ static void uct_cuda_ipc_batch_copy_single(void *dst,
                                                      const void *src,
                                                      size_t size)
{
    typedef int4 vectype;
    const char *s1  = reinterpret_cast<const char*>(src);
    char       *d1  = reinterpret_cast<char *>(dst);
    const vectype *s4;
    vectype *d4;
    int warp, num_warps, idx;
    size_t line, num_lines;

    if (!(align_pow2((intptr_t)s1, sizeof(vectype)) ||
        align_pow2((intptr_t)d1, sizeof(vectype)))) {

        vectype tmp[UNROLL];
        warp      = threadIdx.x / WARP_SIZE;
        num_warps = blockDim.x / WARP_SIZE;
        idx       = threadIdx.x % WARP_SIZE;
        s4        = reinterpret_cast<const vectype*>(s1);
        d4        = reinterpret_cast<vectype*>(d1);
        num_lines = (size / (WARP_SIZE * UNROLL * sizeof(vectype))) *
                    (WARP_SIZE * UNROLL);

        for (line = warp * WARP_SIZE * UNROLL + idx; line < num_lines;
             line += num_warps * WARP_SIZE * UNROLL) {
#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp[i] = ld_global_cg(s4 + (line + WARP_SIZE * i));
            }

#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                st_global_cg(d4 + (line + WARP_SIZE * i), tmp[i]);
            }
        }
        size = size - num_lines * sizeof(vectype);
        if (size == 0) {
            return;
        }

        s4 = s4 + num_lines;
        d4 = d4 + num_lines;
        num_lines = size / sizeof(vectype);
        for (line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vectype v = ld_global_cg(s4 + line);
            st_global_cg(d4 + line, v);
        }

        size = size - num_lines * sizeof(vectype);
        if (size == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + num_lines);
        d1 = reinterpret_cast<char*>(d4 + num_lines);
    }

    for (line = threadIdx.x; line < size; line += blockDim.x) {
        d1[line] = s1[line];
    }
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_cuda_ipc_batch_execute(uct_batch_h tl_batch, uint64_t flags,
                           uint64_t signal_inc, uct_dev_completion_t *comp)
{
    unsigned              lane_id    = threadIdx.x;
    uct_cuda_ipc_batch_t *batch      = (uct_cuda_ipc_batch_t *)tl_batch;
    const int             has_atomic = uct_cuda_ipc_batch_has_atomic(batch);
    size_t                n_vec      = (has_atomic) ? (batch->num - 1) : batch->num;
#if ENABLE_PARAMS_CHECK
    const int has_iov = uct_cuda_ipc_batch_has_iov(batch);
    ucs_status_t status;
#endif

#if ENABLE_PARAMS_CHECK
    status = uct_cuda_ipc_batch_params_check(batch, flags, has_iov, has_atomic, comp);
    if (status != UCS_OK) {
        return status;
    }
#endif

    if (scale != UCT_DEV_SCALE_BLOCK) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (size_t idx = 0 ; idx < n_vec ; idx++) {
#if 1
        uct_cuda_ipc_batch_copy_single<COPY_LOOP_UNROLL>((void *)batch->list[idx].dst,
                                                         (void *)batch->list[idx].src,
                                                         batch->list[idx].size);
#else
        uct_cuda_ipc_batch_copy_single_nv((void *)batch->list[idx].dst,
                                          (void *)batch->list[idx].src,
                                          batch->list[idx].size);
#endif
    }

    __syncthreads();
    if (lane_id == 0) {
        if (has_atomic) {
            uint64_t* p = (uint64_t*)batch->list[batch->num - 1].dst;
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> dst_ref{*p};
            dst_ref.fetch_add(signal_inc, cuda::memory_order_relaxed);
            cuda::atomic_thread_fence(cuda::memory_order_release,
                                      cuda::thread_scope_system);
        }
        comp->count--;
    }
    __syncthreads();
    return UCS_OK;
}

#endif /* UCT_CUDA_IPC_CUH */
