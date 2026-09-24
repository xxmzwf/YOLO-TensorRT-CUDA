#ifndef YOLOTRTCUDA_INFER_H
#define YOLOTRTCUDA_INFER_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace YTC
{
struct ModelInputInfo
{
    int width = 640;
    int height = 640;
    int channels = 3;
    bool fp16 = false;
    bool bakedPreprocess = false;
    bool inputNHWC = false;
    bool dynamicSize = false;
    float inputScale = 1.0f / 255.0f;
};

struct InputTensor
{
    std::array<int64_t, 4> shape = {1, 3, 0, 0};
    bool fp16 = false;
    void* deviceData = nullptr;
    size_t deviceBytes = 0;
    void* sourceDevice = nullptr;
    size_t sourceBytes = 0;
    void* sourceHost = nullptr;
    size_t sourceHostBytes = 0;

    InputTensor() = default;
    InputTensor(const InputTensor&) = delete;
    InputTensor& operator=(const InputTensor&) = delete;
    ~InputTensor();

    void release();
};

struct OutputView
{
    std::vector<int64_t> shape;
    bool fp16 = false;
    bool objectness = false;
    bool yoloxDecode = false;
    bool endToEnd = false;
    bool endToEndSet = false;
    int64_t yoloxDecodeOffset = 0;
    const void* data = nullptr;
    void* hostData = nullptr;
    size_t bytes = 0;
};

class InferEngine
{
public:
    InferEngine();
    ~InferEngine();

    bool loadModel(const std::string& modelPath, int device);
    bool ready() const;
    const ModelInputInfo& inputInfo() const;
    const std::vector<OutputView>& run(InputTensor& input);
    cudaStream_t stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace YTC

#endif // YOLOTRTCUDA_INFER_H
