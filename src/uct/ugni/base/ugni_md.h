/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_CONTEXT_H
#define UCT_UGNI_CONTEXT_H

#include "ugni_types.h"
#include "ugni_def.h"
#include <uct/base/uct_md.h>

/**
 * @breif Static information about UGNI job
 *
 * This is static information about Cray's job.
 * The information is static and does not change since job launch.
 * Therefore, the information is only fetched once.
 */
typedef struct uct_ugni_job_info {
    uint8_t             ptag;                           /**< Protection tag */
    uint32_t            cookie;                         /**< Unique identifier generated by the PMI system */
    int                 pmi_num_of_ranks;               /**< Number of ranks started by PMI */
    int                 pmi_rank_id;                    /**< rank id assigned by PMI */
    int                 num_devices;                    /**< Number of devices */
    uct_ugni_device_t   devices[UCT_UGNI_MAX_DEVICES];  /**< Array of devices */
    bool                initialized;                    /**< Info status */
} uct_ugni_job_info_t;

extern uct_ugni_job_info_t job_info;
extern uct_md_component_t uct_ugni_md_component;

/** @brief Global lock for the component */
extern pthread_mutex_t uct_ugni_global_lock;

/**
 * @brief Helper function to list UGNI resources
 */
uct_ugni_device_t *uct_ugni_device_by_name(const char *dev_name);

static inline gni_nic_device_t uct_ugni_find_gni_device_type(int dev_id)
{
    return job_info.devices[dev_id].type;
}

#endif
