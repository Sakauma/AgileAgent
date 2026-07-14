#include <cstdint>

extern "C" {

// Versioned C ABI used by the Python loader. The inference ABI is enabled only
// after TensorRT engines are generated and validated on the deployment GPU.
std::uint32_t agile_agent_backend_version() { return 1U; }

}
