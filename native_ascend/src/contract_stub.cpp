#include "agile_agent_ascend_backend.h"

#include <cstdlib>
#include <new>
#include <string>

namespace {

struct ContractStubHandle {
  std::string error;
};

const char* kNoHandle =
    "Ascend contract stub: handle is unavailable and CPU fallback is forbidden";

ContractStubHandle* as_handle(void* handle) {
  return static_cast<ContractStubHandle*>(handle);
}

void set_unavailable(ContractStubHandle* handle) {
  if (handle != nullptr) {
    handle->error =
        "Ascend contract stub: CANN runtime is unavailable; backend remains Not Ready";
  }
}

}  // namespace

extern "C" {

uint32_t agile_agent_ascend_backend_version(void) {
  return AGILE_AGENT_ASCEND_ABI_VERSION;
}

void* agile_agent_ascend_create(const char* /*config_json*/) {
  auto* handle = new (std::nothrow) ContractStubHandle();
  set_unavailable(handle);
  return handle;
}

void agile_agent_ascend_destroy(void* handle) {
  delete as_handle(handle);
}

int agile_agent_ascend_ready(void* /*handle*/) {
  return 0;
}

int agile_agent_ascend_warmup(
    void* handle,
    const void* /*encoded_image*/,
    size_t /*encoded_size*/,
    uint32_t /*iterations*/) {
  set_unavailable(as_handle(handle));
  return -1;
}

void* agile_agent_ascend_predict(
    void* handle,
    const void* /*encoded_image*/,
    size_t /*encoded_size*/,
    const char* /*options_json*/) {
  set_unavailable(as_handle(handle));
  return nullptr;
}

void agile_agent_ascend_free_result(void* result) {
  std::free(result);
}

const char* agile_agent_ascend_last_error(void* handle) {
  auto* typed = as_handle(handle);
  return typed == nullptr ? kNoHandle : typed->error.c_str();
}

}  // extern "C"
