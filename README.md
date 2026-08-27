# YoloTrtCuda


YOLO的TensorRT高速目标检测模型推理库，享受极致的预处理推理后处理速度，只依赖TensorRT、CUDA、CUDNN，不依赖OpenCV。
`YoloTrtCuda` 是面向 CUDA/TensorRT 的目标检测库，提供模型转换、推理和后处理能力。
它只包含目标检测模块，不包含分类、分割、姿态或 OBB 模块。

库的公开 C++ 接口、`DetectResultBox`、图像描述结构和调用流程与
[YOLO-ONNXRuntime-DirectML](https://github.com/xxmzwf/YOLO-ONNXRuntime-DirectML) 保持一致，类名替换为
`YoloTrtCuda`。此外提供 `setDevice(int device)` 用于选择 CUDA GPU。

主要特性：

- 使用 TensorRT 执行模型推理；普通输入使用 CUDA 预处理，烘焙输入使用 CPU SIMD resize/letterbox，输出解码和 NMS 使用 CPU SIMD 完成。
- 置信度阈值和 NMS 阈值只在运行时设置，不会嵌入 engine。
- 使用 `tools/onnx_to_yolotrtcuda.py` 生成专用的 FP16 engine 容器。
- 普通 TensorRT `.engine` 文件不会被误加载，必须使用本项目的专用转换脚本。
- 安装时只安装 `YoloTrtCuda` 自身的头文件、动态库、导入库和 CMake 配置，
  不复制 TensorRT、CUDA、cuDNN 的 DLL。

## 支持范围

本项目支持 YOLOv3 至 YOLO26，以及 YOLO-World、YOLOE、YOLO-Master 等变体的部分导出
形式。本项目只处理“原始目标检测输出”路径，因此最终是否可用还取决于 ONNX 是否能导出
为一个单输入、一个或多个兼容的原始检测头。

| 模型类型 | 典型原始输出 | YoloTrtCuda |
| --- | --- | --- |
| YOLOv3–YOLOv7、YOLOv5 风格 | `[1, N, 4+1+C]`，包含 objectness | 支持；自定义类别数时可使用 `--objectness on` |
| YOLOv8、YOLOv9、YOLOv11、YOLOv12 风格 | `[1, 4+C, N]` 或等价的 `[1, N, 4+C]`，不包含 objectness | 支持 |
| YOLOX 风格原始头 | `[1, N, 4+1+C]`，回归值需要按 stride 解码 | 支持；转换脚本会自动识别 `reg_preds/obj_preds/cls_preds`，并使用 YOLOX 解码 |
| YOLOv10、YOLO26 端到端导出 | 常见为 `[1, N, 6]`，NMS 已在模型内部完成 | 不直接支持；需要导出原始检测头 |
| YOLO-World、YOLOE、YOLO-Master 等变体 | 取决于导出图 | 只有在输出符合上述原始检测格式时支持；不包含额外的开放词汇后处理 |

转换脚本会自动将输入烘焙为 `UINT8 NHWC`。烘焙模式将转置、类型转换和归一化放入
TensorRT 图中，缩放和 letterbox 由运行库的 CPU SIMD 路径完成。

以下内容不属于本库范围：

- 分割、姿态、OBB、分类以及包含非检测辅助输出的模型；
- 模型内部已经包含 EfficientNMS 或其他 NMS/阈值后处理的 engine；
- 普通 TensorRT engine；
- 当前运行时的 batch 大于 1 的推理；
- 输入不是单个浮点 NCHW 或本项目烘焙生成的 UINT8 NHWC 图像张量的模型。

类别数可以不是 COCO 的 80 类，但输出布局必须与脚本的 `--objectness` 设置一致。
库只返回数值类别 ID，不读取类别名称文件。

普通 YOLO 原始头的输入预处理默认将像素缩放为 `[0, 1]`。对于自动识别的 YOLOX 原始头，
库使用 YOLOX 常见的 `[0, 255]` 浮点输入，并在 CPU 后处理中执行 stride 为 8、16、32
的网格解码。

## 环境要求

当前 CMake 配置和本机验证环境为：

- Windows x64；
- Visual Studio/MSVC，C++17；
- CMake 3.18 或更高版本；
- CUDA 13.3；
- TensorRT 11.2；
- cuDNN 9.24。

项目默认使用以下路径，但都可以通过 CMake 参数覆盖：

~~~text
D:/Code/Libs/TensorRT-Shared
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3
C:/Program Files/NVIDIA/CUDNN/v9.24
~~~

`CUDNN_PATH` 用于检查/记录开发环境，YoloTrtCuda 本身不将 cuDNN DLL 链接或复制到安装目录。
部署时仍需要在目标机器上自行提供 TensorRT、CUDA、cuDNN 运行时 DLL，并确保它们位于系统
`PATH` 或应用程序可搜索的位置。

## 编译和安装

本项目构建的是动态库。Visual Studio 多配置生成器下，`Release` 通过
`--config Release` 选择，也是本文档的默认构建配置。

在项目根目录执行：

~~~powershell
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DTENSORRT_PATH="D:/Code/Libs/TensorRT-Shared" `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" `
  -DCUDNN_PATH="C:/Program Files/NVIDIA/CUDNN/v9.24"

cmake --build build --config Release --parallel 8
cmake --install build --config Release --prefix "D:/Code/Libs/YoloTrtCuda"
~~~

如果 CMake 没有自动找到 NVCC，可以额外指定：

~~~powershell
-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"
~~~

如果只想为特定 GPU 架构编译，可以在配置时指定架构，例如：

~~~powershell
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES="86" `
  -DTENSORRT_PATH="D:/Code/Libs/TensorRT-Shared" `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" `
  -DCUDNN_PATH="C:/Program Files/NVIDIA/CUDNN/v9.24"
~~~

默认编译的架构为 `75;86;89;90`。如果 CUDA 或 TensorRT 安装在其他位置，只需替换对应的
CMake 参数。

安装目录结构如下：

~~~text
D:/Code/Libs/YoloTrtCuda/
├── bin/
│   └── YoloTrtCuda.dll
├── include/
│   └── YoloTrtCuda.h
└── lib/
    ├── YoloTrtCuda.lib
    └── cmake/YoloTrtCuda/
        ├── YoloTrtCudaConfig.cmake
        ├── YoloTrtCudaConfigVersion.cmake
        ├── YoloTrtCudaTargets.cmake
        └── YoloTrtCudaTargets-release.cmake
~~~

`install` 目录不会包含 `nvinfer*.dll`、`cudart*.dll`、`cudnn*.dll` 等第三方运行时文件。

## 在自己的 CMake 项目中使用

假设安装前缀为 `D:/Code/Libs/YoloTrtCuda`，下游项目可以这样配置：

~~~cmake
list(APPEND CMAKE_PREFIX_PATH "D:/Code/Libs/YoloTrtCuda")

find_package(YoloTrtCuda CONFIG REQUIRED)

add_executable(my_detector main.cpp)
target_link_libraries(my_detector PRIVATE YoloTrtCuda::YoloTrtCuda)

# 将本库 DLL 复制到下游程序旁边。
add_custom_command(TARGET my_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${YoloTrtCuda_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:my_detector>"
    VERBATIM)
~~~

也可以在配置命令中传入安装前缀：

~~~powershell
cmake -S . -B build `
  -DCMAKE_PREFIX_PATH="D:/Code/Libs/YoloTrtCuda"
cmake --build build --config Release
~~~

下游应用还必须自行部署与当前 TensorRT/CUDA/cuDNN 版本匹配的第三方 DLL。它们可以放在
应用程序目录，或者加入系统 `PATH`。本库的 CMake package 不会替下游项目复制这些依赖。

## ONNX 转换为专用 FP16 engine

转换工具位于 [tools/onnx_to_yolotrtcuda.py](tools/onnx_to_yolotrtcuda.py)。它会：

1. 检查 ONNX 是否为单输入、一个或多个 rank-3 原始检测输出；如果同时暴露了最终拼接输出和中间特征头，只保留最终检测输出；
2. 自动将浮点 NCHW 输入烘焙为 UINT8 NHWC，并将转置、Cast、归一化节点加入图中；
3. 将模型转换为 FP16；
4. 使用 TensorRT 构建 engine；
5. 在 engine 外层写入 `YoloTrtCuda` 专用容器头和模型元数据。

对于 YOLOX 风格的原始头，脚本会在元数据中记录 objectness、YOLOX 网格解码和输入像素范围，
运行库加载 engine 后会自动采用对应的预处理和后处理方式。

多个兼容的原始检测输出会全部绑定到 TensorRT，并在运行库中合并候选框后统一执行一次 NMS。
各检测头必须使用相同的类别布局；模型内部的 EfficientNMS 或其他阈值后处理仍不属于本库范围。

因此输出文件不是可以直接交给普通 TensorRT 程序的 `.engine`，也不是普通的
`trtexec --saveEngine` 输出。

### 安装 Python 依赖

需要与本机 Python 版本匹配的 TensorRT Python wheel，以及 `onnx`、`onnxruntime` 和 `numpy`。
下面以 Python 3.11 wheel 为例；如果 Python 版本不同，请选择对应的 wheel：

~~~powershell
py -3 -m pip install onnx onnxruntime numpy
py -3 -m pip install "D:/Code/Libs/TensorRT-Shared/python/tensorrt-11.2.1.2-cp311-none-win_amd64.whl"

# 如果 TensorRT Python 运行时找不到 DLL，可在当前 PowerShell 会话中补充 PATH。
$env:PATH = "D:/Code/Libs/TensorRT-Shared/bin;$env:PATH"
~~~

### 导出并转换

先导出不包含 NMS 的原始 YOLO ONNX。例如使用 Ultralytics：

~~~powershell
yolo export model=models/yolo11n.pt format=onnx batch=1 imgsz=640
~~~

然后运行本项目的转换脚本。只传入 ONNX 文件时，脚本会自动完成模型烘焙和 FP16 转换，
并在同一目录生成 `<原文件名>_fp16_u8.engine`。例如：

~~~powershell
py -3 tools/onnx_to_yolotrtcuda.py models/yolo11n.onnx
~~~

上面的命令会生成 `models/yolo11n_fp16_u8.engine`。脚本会在转换过程中打印阶段进度；
TensorRT 构建 engine 时可能需要等待一段时间。YOLOX 原始头会自动保留其常见的 `0..255`
输入范围，不额外除以 255。已经是 `UINT8 NHWC` 的模型也可以直接传入脚本。

YOLOv5 风格且为自定义类别数的模型，如果脚本无法从输出维度自动判断 objectness，可以显式指定：

~~~powershell
py -3 tools/onnx_to_yolotrtcuda.py models/custom_yolov5.onnx `
  --objectness on
~~~

脚本支持的主要参数：

| 参数 | 说明 |
| --- | --- |
| `INPUT` | 原始或已烘焙的 YOLO ONNX 文件；输出自动保存为同目录下的 `<原文件名>_fp16_u8.engine` |
| `--height`、`--width` | 构建时使用的输入尺寸，默认 640×640 |
| `--objectness auto\|on\|off` | 是否按 YOLOv5 风格解析 objectness，默认 `auto` |
| `--workspace-gb` | TensorRT workspace 上限 |
| `--verbose` | 输出 TensorRT 详细日志 |

本项目故意不使用 EfficientNMS 后处理，因为该类后处理会把阈值和 NMS 配置固化到 engine。
这里的输出解码和 NMS 在 CPU 上执行，NMS 阈值由运行时 API 设置。其他工具生成的普通
TensorRT engine 不能直接作为 YoloTrtCuda 输入。

## API

头文件为 [YoloTrtCuda/include/YoloTrtCuda.h](YoloTrtCuda/include/YoloTrtCuda.h)。

### 公开结构体

~~~cpp
struct DetectResultBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    int classId;
};
~~~

`x`、`y` 是左上角坐标，`width`、`height` 是宽高，坐标单位为原始输入图像像素。

~~~cpp
enum class ImageFormat {
    BGR8,
    RGB8,
    BGRA8,
    RGBA8,
    GRAY8
};

struct ImageView {
    const void* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::size_t stride = 0;
    ImageFormat format = ImageFormat::BGR8;
};
~~~

### 方法说明

| 方法 | 说明 |
| --- | --- |
| `bool setModel(std::string path)` | 加载 YoloTrtCuda 专用 engine；成功返回 `true` |
| `void setDevice(int device)` | 选择 CUDA GPU。建议在 `setModel` 前调用；若模型已加载，切换设备会自动重新加载 |
| `void setConfidenceThreshold(float value)` | 设置运行时置信度阈值 |
| `void setNMSThreshold(float value)` | 设置运行时 IoU NMS 阈值 |
| `void setImage(ImageView& image)` | 设置当前图像视图；库不拥有图像内存 |
| `void preprocess()` | 执行格式转换、缩放和 letterbox；烘焙输入使用 CPU SIMD 生成 UINT8 NHWC |
| `void infer()` | 使用 TensorRT 执行推理 |
| `void postprocess()` | 拷贝输出、解码候选框并执行 NMS |
| `std::vector<DetectResultBox> resultBoxes()` | 获取当前结果的副本 |

典型调用顺序为：

~~~text
setDevice -> setModel -> setImage -> preprocess -> infer -> postprocess -> resultBoxes
~~~

### C++ 示例

下面示例使用 OpenCV 读取 BGR 图像；OpenCV 不是 YoloTrtCuda 的公共依赖，只是示例中的图像来源：

~~~cpp
#include <YoloTrtCuda.h>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main() {
    cv::Mat frame = cv::imread("image.jpg", cv::IMREAD_COLOR);
    if (frame.empty()) {
        return 1;
    }

    YoloTrtCuda detector;
    detector.setDevice(0);

    if (!detector.setModel("models/yolo11n.yolotrtcuda.engine")) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    detector.setConfidenceThreshold(0.40f);
    detector.setNMSThreshold(0.45f);

    ImageView image;
    image.data = frame.data;
    image.width = frame.cols;
    image.height = frame.rows;
    image.channels = frame.channels();
    image.stride = frame.step;
    image.format = ImageFormat::BGR8;

    detector.setImage(image);
    detector.preprocess();
    detector.infer();
    detector.postprocess();

    for (const DetectResultBox& box : detector.resultBoxes()) {
        std::cout << "class=" << box.classId
                  << " score=" << box.score
                  << " box=(" << box.x << ", " << box.y
                  << ", " << box.width << ", " << box.height << ")\n";
    }

    return 0;
}
~~~

`ImageView` 只保存外部图像指针。调用 `preprocess()` 时图像数据必须仍然有效；预处理完成后，
库会把数据复制到自己的 GPU 缓冲区。

## 性能和部署说明

推理过程中 TensorRT 使用独立 CUDA stream；普通输入的预处理使用 CUDA kernel，烘焙输入使用
CPU SIMD resize/letterbox。`infer()` 只包含 TensorRT enqueue 和自身同步，输出 D2H 拷贝、
候选框解码与 NMS 都在 `postprocess()` 中完成。没有在 TensorRT 网络中追加阈值/NMS 层。

因此运行时阈值可调不会改变主 engine，同时避免把可调后处理做成 TensorRT 图的一部分。实际
吞吐和延迟仍取决于 GPU、输入尺寸、模型结构、TensorRT 版本和 batch 配置；本 README 不提供
未经实测的性能数字。

部署时至少需要：

- 应用程序本身；
- `YoloTrtCuda.dll`；
- 与构建版本匹配的 TensorRT、CUDA、cuDNN DLL；
- 使用本项目脚本生成的 `.yolotrtcuda.engine` 文件。

## License

详见 [LICENSE](LICENSE)。
