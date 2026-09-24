#include "YoloTrtCuda.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

namespace YTC
{
namespace
{
void synchronizeStream(const InferEngine& engine)
{
    if (!engine.ready() || engine.stream() == nullptr) {
        return;
    }
    const cudaError_t error = cudaStreamSynchronize(engine.stream());
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(error));
    }
}
}

struct YoloTrtCuda::Impl
{
    InferEngine engine;
    std::string modelPath;
    int device = 0;
    float confidenceThreshold = 0.4f;
    float nmsThreshold = 0.45f;

    ImageView image;
    PreprocessContext preprocessContext;
    PreprocessResult preprocessResult;
    InputTensor inputTensor;
    const std::vector<OutputView>* outputs = nullptr;
    PostprocessContext postprocessContext;
    std::vector<DetectResultBox> results;

    void releaseRuntimeBuffers()
    {
        if (engine.ready()) {
            cudaSetDevice(device);
            if (engine.stream() != nullptr) {
                cudaStreamSynchronize(engine.stream());
            }
        }
        outputs = nullptr;
        inputTensor.release();
        postprocessContext.release();
        results.clear();
    }
};

YoloTrtCuda::YoloTrtCuda()
    : impl(std::make_unique<Impl>())
{
}

YoloTrtCuda::~YoloTrtCuda() = default;

bool YoloTrtCuda::setModel(std::string modelPath)
{
    impl->releaseRuntimeBuffers();
    impl->modelPath = std::move(modelPath);
    return impl->engine.loadModel(impl->modelPath, impl->device);
}

void YoloTrtCuda::setDevice(int device)
{
    if (device == impl->device) {
        return;
    }
    const bool reload = impl->engine.ready();
    impl->releaseRuntimeBuffers();
    impl->device = device;
    if (reload) {
        impl->engine.loadModel(impl->modelPath, device);
    }
}

void YoloTrtCuda::setConfidenceThreshold(float threshold)
{
    impl->confidenceThreshold = threshold;
}

void YoloTrtCuda::setNMSThreshold(float threshold)
{
    impl->nmsThreshold = threshold;
}

void YoloTrtCuda::setImage(ImageView& image)
{
    impl->image = image;
    impl->outputs = nullptr;
}

void YoloTrtCuda::preprocess()
{
    impl->outputs = nullptr;
    if (!impl->engine.ready() || impl->image.data == nullptr ||
        impl->image.width <= 0 || impl->image.height <= 0) {
        return;
    }
    impl->preprocessResult = YTC::preprocess(
        impl->engine.inputInfo(),
        impl->image,
        impl->preprocessContext,
        impl->inputTensor,
        impl->engine.stream());
    synchronizeStream(impl->engine);
}

void YoloTrtCuda::infer()
{
    impl->outputs = nullptr;
    if (!impl->engine.ready() || impl->inputTensor.deviceData == nullptr) {
        return;
    }
    try {
        impl->outputs = &impl->engine.run(impl->inputTensor);
        synchronizeStream(impl->engine);
    } catch (const std::exception& exception) {
        std::cerr << "[YoloTrtCuda] inference failed: " << exception.what() << std::endl;
    }
}

void YoloTrtCuda::postprocess()
{
    impl->results.clear();
    if (impl->outputs == nullptr) {
        return;
    }
    YTC::postprocess(
        *impl->outputs,
        impl->preprocessResult,
        impl->confidenceThreshold,
        impl->nmsThreshold,
        impl->postprocessContext,
        impl->results,
        impl->engine.stream());
}

std::vector<DetectResultBox> YoloTrtCuda::resultBoxes()
{
    return impl->results;
}
} // namespace YTC
