#ifndef AGILE_AGENT_ASCEND_BACKEND_H
#define AGILE_AGENT_ASCEND_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(AGILE_AGENT_ASCEND_BUILD)
#define AGILE_AGENT_ASCEND_API __declspec(dllexport)
#else
#define AGILE_AGENT_ASCEND_API __declspec(dllimport)
#endif
#else
#define AGILE_AGENT_ASCEND_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Version 1 is a whole-Agent ABI: one handle owns the base detector,
 * every active incremental detector, Scene-SensorNet, DVPP/VPC/AIPP state,
 * streams and reusable ring buffers. A request must execute every class owner;
 * the implementation may not route by filename, labels or test membership.
 */
#define AGILE_AGENT_ASCEND_ABI_VERSION 1u

AGILE_AGENT_ASCEND_API uint32_t agile_agent_ascend_backend_version(void);

/*
 * config_json contains device_id, fixed OM paths/shapes, scheduling mode,
 * ring_slots and frozen fusion settings. Creation must leave the handle Not
 * Ready if any OM or CANN resource is unavailable. CPU model fallback is
 * forbidden.
 */
AGILE_AGENT_ASCEND_API void* agile_agent_ascend_create(const char* config_json);
AGILE_AGENT_ASCEND_API void agile_agent_ascend_destroy(void* handle);
AGILE_AGENT_ASCEND_API int agile_agent_ascend_ready(void* handle);

/* Warm up the complete PNG-to-result path with a real encoded image. */
AGILE_AGENT_ASCEND_API int agile_agent_ascend_warmup(
    void* handle,
    const void* encoded_image,
    size_t encoded_size,
    uint32_t iterations);

/*
 * Returns an allocated UTF-8 JSON result or NULL on failure. The result must
 * include final detections, soft context output and segmented timings for
 * decode, preprocess, models, postprocess and total wall time.
 */
AGILE_AGENT_ASCEND_API void* agile_agent_ascend_predict(
    void* handle,
    const void* encoded_image,
    size_t encoded_size,
    const char* options_json);

AGILE_AGENT_ASCEND_API void agile_agent_ascend_free_result(void* result);
AGILE_AGENT_ASCEND_API const char* agile_agent_ascend_last_error(void* handle);

#ifdef __cplusplus
}
#endif

#endif
