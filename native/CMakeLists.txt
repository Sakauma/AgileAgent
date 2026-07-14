cmake_minimum_required(VERSION 3.24)
project(agile_agent_tensorrt LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)

add_library(agile_agent_tensorrt SHARED src/backend.cpp)
target_compile_features(agile_agent_tensorrt PRIVATE cxx_std_17)
target_include_directories(agile_agent_tensorrt PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(agile_agent_tensorrt PRIVATE CUDA::cudart ${TENSORRT_LIBRARY} ${OpenCV_LIBS})
