/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_EP_DEV_H
#define UCT_CUDA_IPC_EP_DEV_H

#include <uct/api/cuda/uct.h>
#include <uct/api/uct_def.h>
#include <uct/api/uct.h>

#define UCT_DEV_TL_CUDA_IPC 2

#define UCT_CUDA_IPC_PUT 0
#define UCT_CUDA_IPC_GET 1
#define UCT_CUDA_IPC_ATOMIC_FA 2

typedef struct {
    int e_op;
    size_t size;
    uint64_t src;
    uint32_t lkey;
    uint64_t dst;
    uint32_t rkey;
} uct_cuda_ipc_batch_elem_t;

typedef struct {
    uct_batch_t super;
    int op;
    size_t num;
    uint64_t atomic_buff;
    uct_cuda_ipc_batch_elem_t list[0];
} uct_cuda_ipc_batch_t;

#endif