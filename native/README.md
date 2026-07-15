# TensorRT 原生后端

此目录定义灵动Agent的C++/CUDA后端ABI。构建需要CUDA、TensorRT、OpenCV开发包以及与部署GPU匹配的FP16 engine：

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release -j
```

原生库实现版本化C ABI、OpenCV内存解码、YOLO letterbox、动态batch、可复用CUDA缓冲区、TensorRT前向和class-aware NMS。Python继续负责策略、代际、审计与Web协议。

构建成功不代表可以直接上线。必须先在目标GPU上导出匹配的engine，再使用TensorRT验收命令核对CUDA基线精度、API延迟、批量吞吐和并发稳定性；门禁未通过时配置会保持原后端，不会回退到CPU。
