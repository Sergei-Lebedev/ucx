/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc.inl"
#include "cuda_ipc_ep_dev.h"

#include <uct/api/cuda/uct.h>
#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>

static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_cuda_ipc_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->remote_pid = *(const pid_t*)params->iface_addr;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);

#define uct_cuda_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))

int uct_cuda_ipc_ep_is_connected(const uct_ep_h tl_ep,
                                 const uct_ep_is_connected_params_t *params)
{
    const uct_cuda_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_cuda_ipc_ep_t);

    if (!uct_base_ep_is_connected(tl_ep, params)) {
        return 0;
    }

    return ep->remote_pid == *(pid_t*)params->iface_addr;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_ipc_ctx_rsc_get(
        uct_cuda_ipc_iface_t *iface, uct_cuda_ipc_ctx_rsc_t **ctx_rsc_p)
{
    unsigned long long ctx_id;
    ucs_status_t status;
    CUresult result;
    uct_cuda_ctx_rsc_t *ctx_rsc;

    result = uct_cuda_base_ctx_get_id(NULL, &ctx_id);
    if (ucs_unlikely(result != CUDA_SUCCESS)) {
        UCT_CUDADRV_LOG(cuCtxGetId, UCS_LOG_LEVEL_ERROR, result);
        return UCS_ERR_IO_ERROR;
    }

    status = uct_cuda_base_ctx_rsc_get(&iface->super, ctx_id, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *ctx_rsc_p = ucs_derived_of(ctx_rsc, uct_cuda_ipc_ctx_rsc_t);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_post_cuda_async_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  uct_completion_t *comp, int direction)
{
    uct_cuda_ipc_iface_t *iface       = ucs_derived_of(tl_ep->iface,
                                                       uct_cuda_ipc_iface_t);
    uct_cuda_ipc_unpacked_rkey_t *key = (uct_cuda_ipc_unpacked_rkey_t *)rkey;
    CUdevice cuda_device;
    int is_ctx_pushed;
    void *mapped_rem_addr;
    void *mapped_addr;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    uct_cuda_ipc_ctx_rsc_t *ctx_rsc;
    uct_cuda_queue_desc_t *q_desc;
    ucs_status_t status;
    CUdeviceptr dst, src;
    CUcontext UCS_V_UNUSED cuda_context;
    CUstream *stream;
    size_t offset;

    if (ucs_unlikely(0 == iov[0].length)) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)iov[0].buffer,
                                             &cuda_device, &is_ctx_pushed);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    status = uct_cuda_ipc_map_memhandle(&key->super, cuda_device, &mapped_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    status = uct_cuda_ipc_ctx_rsc_get(iface, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    offset          = (uintptr_t)remote_addr - (uintptr_t)key->super.d_bptr;
    mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
    ucs_assert(offset <= key->super.b_len);

    /* round-robin */
    q_desc = &ctx_rsc->queue_desc[key->stream_id % iface->config.max_streams];
    stream = &q_desc->stream;
    status = uct_cuda_base_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    if (ucs_unlikely(stream == NULL)) {
        ucs_error("stream=%d for dev_num=%d not available", key->stream_id,
                  cuda_device);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    cuda_ipc_event = ucs_mpool_get(&ctx_rsc->super.event_mp);
    if (ucs_unlikely(cuda_ipc_event == NULL)) {
        ucs_error("Failed to allocate cuda_ipc event object");
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    dst = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? mapped_rem_addr : iov[0].buffer);
    src = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? iov[0].buffer : mapped_rem_addr);

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyDtoDAsync(dst, src, iov[0].length,
                                                        *stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuEventRecord(cuda_ipc_event->super.event,
                                                    *stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        goto out;
    }

    if (ucs_queue_is_empty(&q_desc->event_queue)) {
        ucs_queue_push(&iface->super.active_queue, &q_desc->queue);
    }

    ucs_queue_push(&q_desc->event_queue, &cuda_ipc_event->super.queue);
    cuda_ipc_event->super.comp  = comp;
    cuda_ipc_event->mapped_addr = mapped_addr;
    cuda_ipc_event->d_bptr      = (uintptr_t)key->super.d_bptr;
    cuda_ipc_event->pid         = key->super.pid;
    cuda_ipc_event->cuda_device = cuda_device;
    ucs_trace("cuMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             cuda_ipc_event, (void *) dst, (void *) src, iov[0].length);
    status = UCS_INPROGRESS;

out:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_GET);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_PUT);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_cuda_ipc_ep_batch_prepare(uct_ep_h tl_ep, const uct_rma_iov_t *iov,
                                           size_t iovcnt, uint64_t signal_va,
                                           uct_rkey_t signal_rkey, uct_batch_h *batch_p)
{
    size_t batch_size;
    uct_cuda_ipc_batch_t *batch, *batch_gpu;
    int has_signal = (signal_va != 0);
    size_t batch_num = iovcnt + (has_signal ? 1 : 0);
    void *mapped_addr, *mapped_rem_addr;
    ucs_status_t status;
    CUresult cerr;
    uct_cuda_ipc_unpacked_rkey_t *key;
    size_t offset;
    CUdevice cuda_device;
    int is_ctx_pushed;

    /* assume all VAs are on the same device,
       otherwise we need to push ctx in the loop */
    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)iov[0].local_va,
                                             &cuda_device, &is_ctx_pushed);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    batch_size = sizeof(uct_cuda_ipc_batch_t) +
                 batch_num * sizeof(uct_cuda_ipc_batch_elem_t);
    batch = ucs_calloc(1, batch_size, "cuda ipc batch");
    if (batch == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    cerr = cuMemAlloc((CUdeviceptr*)&batch_gpu, batch_size);
    if (cerr != CUDA_SUCCESS) {
        ucs_error("cuMemAlloc failed: %s",
                  uct_cuda_base_cu_get_error_string(cerr));
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    batch->super.tl_id = UCT_DEV_TL_CUDA_IPC;
    batch->num = batch_num;
    batch->op = UCT_CUDA_IPC_PUT;

    for (size_t i = 0; i < iovcnt; i++) {
        key = (uct_cuda_ipc_unpacked_rkey_t *)iov[i].rkey;
        status = uct_cuda_ipc_map_memhandle(&key->super, cuda_device,
                                            &mapped_addr);
        if (ucs_unlikely(status != UCS_OK)) {
            ucs_error("failed to map memhandle: %d", status);
            goto err;
        }

        offset = (uintptr_t)iov[i].remote_va - (uintptr_t)key->super.d_bptr;
        mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
        batch->list[i].e_op = batch->op;
        batch->list[i].size = iov[i].length;
        batch->list[i].src = (uint64_t)iov[i].local_va;
        batch->list[i].dst = (uint64_t)mapped_rem_addr;
    }

    if (has_signal) {
        key = (uct_cuda_ipc_unpacked_rkey_t *)signal_rkey;
        status = uct_cuda_ipc_map_memhandle(&key->super, cuda_device,
                                            &mapped_addr);
        if (ucs_unlikely(status != UCS_OK)) {
            ucs_error("failed to map memhandle: %d", status);
            goto err;
        }
        offset = (uintptr_t)signal_va - (uintptr_t)key->super.d_bptr;
        mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
        batch->list[iovcnt].e_op = UCT_CUDA_IPC_ATOMIC_FA;
        batch->list[iovcnt].size = sizeof(uint64_t);
        batch->list[iovcnt].src = (uint64_t)&batch_gpu->atomic_buff;
        batch->list[iovcnt].dst = (uint64_t)mapped_rem_addr;
    }

    cerr = cuMemcpyHtoD((CUdeviceptr)batch_gpu, batch, batch_size);
    if (cerr != CUDA_SUCCESS) {
        ucs_error("cuMemcpyHtoD failed: %s",
                  uct_cuda_base_cu_get_error_string(cerr));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }
    *batch_p = &batch_gpu->super;
    status = UCS_OK;
    goto out;

err:
    cuMemFree((CUdeviceptr)batch_gpu);
out:
    ucs_free(batch);
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

void uct_cuda_ipc_ep_batch_release(uct_ep_h tl_ep, uct_batch_h batch)
{
    cuMemFree((CUdeviceptr)batch);
}

ucs_status_t uct_cuda_ipc_ep_export_dev(uct_ep_h tl_ep, uct_dev_ep_h *dev_ep_p)
{
    return UCS_OK;
}