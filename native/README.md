# TensorRT 原生后端

此目录定义灵动Agent的C++/CUDA后端ABI。构建需要CUDA、TensorRT、OpenCV开发包以及与部署GPU匹配的FP16 engine：

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release -j
```

当前production使用Python策略层调用已验收的TensorRT engine，尚未切换到本目录的原生ABI。Python加载器采用严格门禁：原生库、基础检测engine或场景engine缺失时立即失败，不会回退到CPU。只有补齐解码、预处理、CUDA stream、NMS和结构化结果ABI，并通过95张lock-val精度与性能验收后，才能把`inference.backend`切换为`tensorrt_native`。
