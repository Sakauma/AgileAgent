#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) last_message = message == nullptr ? "TensorRT error" : message;
  }
  std::string last_message;
};

struct Detection {
  float x1{};
  float y1{};
  float x2{};
  float y2{};
  float confidence{};
  int class_id{};
};

struct LetterboxInfo {
  float scale{};
  float pad_x{};
  float pad_y{};
  int width{};
  int height{};
};

float iou(const Detection& a, const Detection& b) {
  const float intersection = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                             std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  const float denominator = area_a + area_b - intersection;
  return denominator > 0.0F ? intersection / denominator : 0.0F;
}

std::vector<Detection> class_aware_nms(std::vector<Detection> rows, float threshold, int max_det) {
  std::sort(rows.begin(), rows.end(), [](const Detection& a, const Detection& b) {
    return a.confidence > b.confidence;
  });
  std::vector<Detection> kept;
  for (const auto& candidate : rows) {
    bool suppressed = false;
    for (const auto& existing : kept) {
      if (candidate.class_id == existing.class_id && iou(candidate, existing) >= threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(candidate);
    if (static_cast<int>(kept.size()) >= max_det) break;
  }
  return kept;
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open TensorRT engine: " + path);
  const auto size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("cannot read TensorRT engine: " + path);
  }
  if (bytes.size() > sizeof(std::int32_t)) {
    std::int32_t metadata_size = 0;
    std::memcpy(&metadata_size, bytes.data(), sizeof(metadata_size));
    const std::size_t engine_offset = sizeof(metadata_size) + static_cast<std::size_t>(std::max(metadata_size, 0));
    if (metadata_size > 0 && engine_offset < bytes.size()) {
      try {
        json::parse(bytes.begin() + sizeof(metadata_size), bytes.begin() + engine_offset);
        return {bytes.begin() + engine_offset, bytes.end()};
      } catch (const json::parse_error&) {
        // Plain TensorRT plans have no Ultralytics metadata prefix.
      }
    }
  }
  return bytes;
}

std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t value = 1;
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) throw std::runtime_error("unresolved TensorRT tensor shape");
    value *= static_cast<std::size_t>(dims.d[index]);
  }
  return value;
}

class Backend {
 public:
  explicit Backend(const json& config) {
    const auto engine_bytes = read_binary(config.at("detector_engine").get<std::string>());
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    engine.reset(runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed: " + logger.last_message);
    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("createExecutionContext failed");
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
      const char* name = engine->getIOTensorName(index);
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
      else output_name = name;
    }
    if (input_name.empty() || output_name.empty()) throw std::runtime_error("detector engine IO contract invalid");
    const auto input_dims = engine->getTensorShape(input_name.c_str());
    if (input_dims.nbDims != 4 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0) {
      throw std::runtime_error("detector engine must use NCHW with fixed spatial dimensions");
    }
    input_height = input_dims.d[2];
    input_width = input_dims.d[3];
    if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("cudaStreamCreate failed");
  }

  ~Backend() {
    release_buffers();
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  int warmup() {
    cv::Mat image(input_height, input_width, CV_8UC3, cv::Scalar(0, 0, 0));
    infer({image}, 0.5F, 0.7F, 300);
    return 0;
  }

  json predict(const std::vector<cv::Mat>& images, const json& options) {
    const float confidence = options.value("conf", 0.5F);
    const float nms_iou = options.value("iou", 0.7F);
    const int max_det = options.value("max_det", 300);
    return infer(images, confidence, nms_iou, max_det);
  }

  std::string last_error;

 private:
  template <typename T>
  struct TrtDelete {
    void operator()(T* object) const { delete object; }
  };

  void release_buffers() {
    if (device_input != nullptr) cudaFree(device_input);
    if (device_output != nullptr) cudaFree(device_output);
    device_input = nullptr;
    device_output = nullptr;
    input_capacity = 0;
    output_capacity = 0;
  }

  void ensure_buffer(void** pointer, std::size_t& capacity, std::size_t bytes) {
    if (bytes <= capacity) return;
    if (*pointer != nullptr) cudaFree(*pointer);
    if (cudaMalloc(pointer, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    capacity = bytes;
  }

  json infer(const std::vector<cv::Mat>& images, float confidence, float nms_iou, int max_det) {
    if (images.empty()) return json{{"results", json::array()}, {"timings", json::object()}};
    std::lock_guard<std::mutex> guard(mutex);
    const auto preprocessing_started = Clock::now();
    const int batch = static_cast<int>(images.size());
    const std::size_t plane = static_cast<std::size_t>(input_height) * input_width;
    std::vector<float> input(static_cast<std::size_t>(batch) * 3 * plane);
    std::vector<LetterboxInfo> letterboxes;
    letterboxes.reserve(images.size());
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
      const cv::Mat& source = images[batch_index];
      const float scale = std::min(static_cast<float>(input_width) / source.cols,
                                   static_cast<float>(input_height) / source.rows);
      const int resized_width = std::max(1, static_cast<int>(std::round(source.cols * scale)));
      const int resized_height = std::max(1, static_cast<int>(std::round(source.rows * scale)));
      const int pad_x = (input_width - resized_width) / 2;
      const int pad_y = (input_height - resized_height) / 2;
      cv::Mat resized;
      cv::resize(source, resized, cv::Size(resized_width, resized_height));
      cv::Mat canvas(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
      resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
      cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
      const std::size_t offset = static_cast<std::size_t>(batch_index) * 3 * plane;
      for (int y = 0; y < input_height; ++y) {
        const auto* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_width; ++x) {
          for (int channel = 0; channel < 3; ++channel) {
            input[offset + static_cast<std::size_t>(channel) * plane + static_cast<std::size_t>(y) * input_width + x] =
                static_cast<float>(row[x][channel]) / 255.0F;
          }
        }
      }
      letterboxes.push_back({scale, static_cast<float>(pad_x), static_cast<float>(pad_y), source.cols, source.rows});
    }
    nvinfer1::Dims4 input_dims{batch, 3, input_height, input_width};
    if (!context->setInputShape(input_name.c_str(), input_dims)) throw std::runtime_error("setInputShape failed");
    const auto output_dims = context->getTensorShape(output_name.c_str());
    const auto input_bytes = input.size() * sizeof(float);
    const auto output_elements = volume(output_dims);
    const auto output_type = engine->getTensorDataType(output_name.c_str());
    const std::size_t output_element_bytes = output_type == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);
    const auto output_bytes = output_elements * output_element_bytes;
    ensure_buffer(&device_input, input_capacity, input_bytes);
    ensure_buffer(&device_output, output_capacity, output_bytes);
    const auto preprocessing_finished = Clock::now();

    if (cudaMemcpyAsync(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("input cudaMemcpyAsync failed");
    }
    context->setTensorAddress(input_name.c_str(), device_input);
    context->setTensorAddress(output_name.c_str(), device_output);
    const auto inference_started = Clock::now();
    if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
    std::vector<float> output(output_elements);
    if (output_type == nvinfer1::DataType::kHALF) {
      std::vector<__half> temporary(output_elements);
      cudaMemcpyAsync(temporary.data(), device_output, output_bytes, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      std::transform(temporary.begin(), temporary.end(), output.begin(),
                     [](const __half value) { return __half2float(value); });
    } else {
      cudaMemcpyAsync(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }
    const auto inference_finished = Clock::now();

    if (output_dims.nbDims != 3) throw std::runtime_error("YOLO output must be rank 3");
    const bool channels_first = output_dims.d[1] < output_dims.d[2];
    const int channels = channels_first ? output_dims.d[1] : output_dims.d[2];
    const int anchors = channels_first ? output_dims.d[2] : output_dims.d[1];
    const int classes = channels - 4;
    if (classes <= 0) throw std::runtime_error("YOLO output class dimension invalid");
    auto value_at = [&](int b, int channel, int anchor) {
      if (channels_first) return output[(static_cast<std::size_t>(b) * channels + channel) * anchors + anchor];
      return output[(static_cast<std::size_t>(b) * anchors + anchor) * channels + channel];
    };
    json results = json::array();
    for (int b = 0; b < batch; ++b) {
      std::vector<Detection> candidates;
      const auto& transform = letterboxes[b];
      for (int anchor = 0; anchor < anchors; ++anchor) {
        int best_class = 0;
        float best_score = value_at(b, 4, anchor);
        for (int class_id = 1; class_id < classes; ++class_id) {
          const float score = value_at(b, 4 + class_id, anchor);
          if (score > best_score) {
            best_score = score;
            best_class = class_id;
          }
        }
        if (best_score < confidence) continue;
        const float center_x = value_at(b, 0, anchor);
        const float center_y = value_at(b, 1, anchor);
        const float width = value_at(b, 2, anchor);
        const float height = value_at(b, 3, anchor);
        Detection detection;
        detection.x1 = std::clamp((center_x - width / 2.0F - transform.pad_x) / transform.scale, 0.0F, static_cast<float>(transform.width));
        detection.y1 = std::clamp((center_y - height / 2.0F - transform.pad_y) / transform.scale, 0.0F, static_cast<float>(transform.height));
        detection.x2 = std::clamp((center_x + width / 2.0F - transform.pad_x) / transform.scale, 0.0F, static_cast<float>(transform.width));
        detection.y2 = std::clamp((center_y + height / 2.0F - transform.pad_y) / transform.scale, 0.0F, static_cast<float>(transform.height));
        detection.confidence = best_score;
        detection.class_id = best_class;
        candidates.push_back(detection);
      }
      const auto kept = class_aware_nms(std::move(candidates), nms_iou, max_det);
      json detections = json::array();
      for (const auto& detection : kept) {
        detections.push_back({
            {"class_id", detection.class_id}, {"confidence", detection.confidence},
            {"xyxy", {detection.x1, detection.y1, detection.x2, detection.y2}},
        });
      }
      results.push_back({{"detections", detections}});
    }
    const auto postprocessing_finished = Clock::now();
    const auto elapsed = [](Clock::time_point start, Clock::time_point finish) {
      return std::chrono::duration<double, std::milli>(finish - start).count();
    };
    const json timings = {
        {"preprocess", elapsed(preprocessing_started, preprocessing_finished) / batch},
        {"inference", elapsed(inference_started, inference_finished) / batch},
        {"postprocess", elapsed(inference_finished, postprocessing_finished) / batch},
    };
    for (auto& result : results) result["timings"] = timings;
    return {{"results", results}, {"timings", timings}};
  }

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context;
  std::string input_name;
  std::string output_name;
  int input_width{};
  int input_height{};
  cudaStream_t stream{};
  void* device_input{};
  void* device_output{};
  std::size_t input_capacity{};
  std::size_t output_capacity{};
  std::mutex mutex;
};

thread_local std::string global_error;

char* copy_result(const std::string& value) {
  auto* output = new char[value.size() + 1];
  std::memcpy(output, value.c_str(), value.size() + 1);
  return output;
}

}  // namespace

extern "C" {

std::uint32_t agile_agent_backend_version() { return 1U; }

void* agile_agent_create(const char* config_json) {
  try {
    if (config_json == nullptr) throw std::runtime_error("native backend config is null");
    return new Backend(json::parse(config_json));
  } catch (const std::exception& error) {
    global_error = error.what();
    return nullptr;
  }
}

void agile_agent_destroy(void* handle) { delete static_cast<Backend*>(handle); }

int agile_agent_warmup(void* handle) {
  try {
    if (handle == nullptr) throw std::runtime_error("native backend handle is null");
    return static_cast<Backend*>(handle)->warmup();
  } catch (const std::exception& error) {
    global_error = error.what();
    if (handle != nullptr) static_cast<Backend*>(handle)->last_error = error.what();
    return -1;
  }
}

char* agile_agent_predict_batch(
    void* handle, const void* const* data, const std::size_t* sizes,
    std::size_t count, const char* options_json) {
  try {
    if (handle == nullptr || data == nullptr || sizes == nullptr || count == 0) {
      throw std::runtime_error("native batch arguments are invalid");
    }
    std::vector<cv::Mat> images;
    images.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const auto* begin = static_cast<const std::uint8_t*>(data[index]);
      cv::Mat encoded(1, static_cast<int>(sizes[index]), CV_8UC1, const_cast<std::uint8_t*>(begin));
      cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
      if (image.empty()) throw std::runtime_error("OpenCV failed to decode input image");
      images.push_back(std::move(image));
    }
    const json options = options_json == nullptr ? json::object() : json::parse(options_json);
    return copy_result(static_cast<Backend*>(handle)->predict(images, options).dump());
  } catch (const std::exception& error) {
    global_error = error.what();
    if (handle != nullptr) static_cast<Backend*>(handle)->last_error = error.what();
    return nullptr;
  }
}

void agile_agent_free_result(char* result) { delete[] result; }

const char* agile_agent_last_error(void* handle) {
  if (handle != nullptr && !static_cast<Backend*>(handle)->last_error.empty()) {
    return static_cast<Backend*>(handle)->last_error.c_str();
  }
  return global_error.c_str();
}

}  // extern "C"
