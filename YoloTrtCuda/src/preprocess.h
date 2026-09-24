#ifndef YOLOTRTCUDA_PREPROCESS_H
#define YOLOTRTCUDA_PREPROCESS_H

#include <vector>

#include <cuda_runtime_api.h>

#include "YoloTrtCuda.h"
#include "infer.h"

namespace YTC
{
struct PreprocessResult
{
    int imageWidth = 0;
    int imageHeight = 0;
    int modelWidth = 0;
    int modelHeight = 0;
    float invScale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};

struct PreprocessContext
{
    int sourceWidth = 0;
    int sourceHeight = 0;
    size_t sourceStride = 0;
    int xSafe = 0;
    std::vector<int> xOffset0;
    std::vector<int> xOffset1;
    std::vector<float> xWeight;
    std::vector<float> rowA;
    std::vector<float> rowB;
};

PreprocessResult preprocess(
    const ModelInputInfo& info,
    const ImageView& image,
    PreprocessContext& context,
    InputTensor& tensor,
    cudaStream_t stream);

void launchLetterbox(
    const void* source,
    int sourceWidth,
    int sourceHeight,
    size_t sourceStride,
    void* destination,
    int destinationWidth,
    int destinationHeight,
    int sourceFormat,
    int modelChannels,
    bool outputFp16,
    float inputScale,
    int validWidth,
    int validHeight,
    int offsetX,
    int offsetY,
    cudaStream_t stream);
} // namespace YTC

#endif // YOLOTRTCUDA_PREPROCESS_H
