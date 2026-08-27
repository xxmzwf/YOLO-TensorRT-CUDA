#include "infer.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <NvInferRuntime.h>

#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace
{
constexpr char kEngineMagic[16] = {
    'Y', 'o', 'l', 'o', 'T', 'r', 't', 'C', 'u', 'd', 'a', 'V', '1', 0, 0, 0};
constexpr uint32_t kEngineVersion = 1;

#pragma pack(push, 1)
struct EngineHeader
{
    char magic[16];
    uint32_t version;
    uint32_t metadataSize;
    uint64_t engineSize;
};
#pragma pack(pop)

struct EngineContainer
{
    std::vector<uint8_t> serialized;
    bool objectness = false;
    bool objectnessSet = false;
    bool yoloxDecode = false;
    bool yoloxDecodeSet = false;
    bool endToEnd = false;
    bool endToEndSet = false;
    bool rawInput = false;
    bool rawInputSet = false;
    bool bakedPreprocess = false;
    bool bakedPreprocessSet = false;
};

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

bool readMetadataBoolean(const std::string& metadata, const char* key, bool& value)
{
    const std::string quotedKey = std::string("\"") + key + "\"";
    const size_t keyPosition = metadata.find(quotedKey);
    if (keyPosition == std::string::npos) {
        return false;
    }
    size_t valuePosition = metadata.find(':', keyPosition + quotedKey.size());
    if (valuePosition == std::string::npos) {
        return false;
    }
    ++valuePosition;
    while (valuePosition < metadata.size() &&
           std::isspace(static_cast<unsigned char>(metadata[valuePosition]))) {
        ++valuePosition;
    }
    if (metadata.compare(valuePosition, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (metadata.compare(valuePosition, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

EngineContainer readEngineContainer(const std::string& path)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Unable to open engine: " + path);
    }

    const std::streamsize fileSize = file.tellg();
    if (fileSize < static_cast<std::streamsize>(sizeof(EngineHeader))) {
        throw std::runtime_error("The file is not a YoloTrtCuda engine: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> fileData(static_cast<size_t>(fileSize));
    if (!file.read(reinterpret_cast<char*>(fileData.data()), fileSize)) {
        throw std::runtime_error("Unable to read engine: " + path);
    }

    EngineHeader header{};
    std::memcpy(&header, fileData.data(), sizeof(header));
    if (std::memcmp(header.magic, kEngineMagic, sizeof(kEngineMagic)) != 0 ||
        header.version != kEngineVersion) {
        throw std::runtime_error("The file is not a supported YoloTrtCuda engine: " + path);
    }

    const uint64_t payloadOffset = static_cast<uint64_t>(sizeof(EngineHeader)) + header.metadataSize;
    if (payloadOffset > fileData.size() || header.engineSize > fileData.size() - payloadOffset) {
        throw std::runtime_error("The YoloTrtCuda engine container is truncated: " + path);
    }

    EngineContainer container;
    const std::string metadata(
        reinterpret_cast<const char*>(fileData.data() + sizeof(EngineHeader)),
        header.metadataSize);
    container.objectnessSet = readMetadataBoolean(metadata, "objectness", container.objectness);
    container.yoloxDecodeSet = readMetadataBoolean(
        metadata, "yolox_decode", container.yoloxDecode);
    container.endToEndSet = readMetadataBoolean(
        metadata, "end_to_end", container.endToEnd);
    container.rawInputSet = readMetadataBoolean(metadata, "raw_input", container.rawInput);
    container.bakedPreprocessSet = readMetadataBoolean(
        metadata, "preprocess_baked", container.bakedPreprocess);
    container.serialized = std::vector<uint8_t>(
        fileData.begin() + static_cast<std::ptrdiff_t>(payloadOffset),
        fileData.begin() + static_cast<std::ptrdiff_t>(payloadOffset + header.engineSize));
    return container;
}

bool isDynamic(const nvinfer1::Dims& shape)
{
    for (int i = 0; i < shape.nbDims; ++i) {
        if (shape.d[i] < 0) {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> toVector(const nvinfer1::Dims& shape)
{
    std::vector<int64_t> result;
    result.reserve(shape.nbDims);
    for (int i = 0; i < shape.nbDims; ++i) {
        result.push_back(shape.d[i]);
    }
    return result;
}

class TrtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[YoloTrtCuda] " << message << std::endl;
        }
    }
};
}

InputTensor::~InputTensor()
{
    release();
}

void InputTensor::release()
{
    if (deviceData != nullptr) {
        cudaFree(deviceData);
        deviceData = nullptr;
    }
    if (sourceDevice != nullptr) {
        cudaFree(sourceDevice);
        sourceDevice = nullptr;
    }
    if (sourceHost != nullptr) {
        cudaFreeHost(sourceHost);
        sourceHost = nullptr;
    }
    deviceBytes = 0;
    sourceBytes = 0;
    sourceHostBytes = 0;
}

struct InferEngine::Impl
{
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    cudaGraph_t inferenceGraph = nullptr;
    cudaGraphExec_t inferenceGraphExec = nullptr;
    void* graphInput = nullptr;
    int device = 0;
    bool sessionReady = false;
    ModelInputInfo info;
    std::string inputName;

    struct OutputBuffer
    {
        std::string name;
        std::vector<int64_t> shape;
        bool fp16 = false;
        bool objectness = false;
        bool yoloxDecode = false;
        bool endToEnd = false;
        bool endToEndSet = false;
        int64_t yoloxDecodeOffset = 0;
        void* device = nullptr;
        void* host = nullptr;
        size_t bytes = 0;

        void release()
        {
            if (device != nullptr) {
                cudaFree(device);
                device = nullptr;
            }
            if (host != nullptr) {
                cudaFreeHost(host);
                host = nullptr;
            }
            bytes = 0;
        }
    };

    std::vector<OutputBuffer> outputBuffers;
    std::vector<OutputView> views;

    void resetInferenceGraph()
    {
        if (inferenceGraphExec != nullptr) {
            cudaGraphExecDestroy(inferenceGraphExec);
            inferenceGraphExec = nullptr;
        }
        if (inferenceGraph != nullptr) {
            cudaGraphDestroy(inferenceGraph);
            inferenceGraph = nullptr;
        }
        graphInput = nullptr;
    }

    void captureInferenceGraph(void* inputData)
    {
        resetInferenceGraph();

        // TensorRT-YOLO performs one regular enqueue before capture so that
        // TensorRT has completed any first-use initialization outside the
        // graph. The captured graph then contains only the steady-state
        // enqueue path and is replayed for every subsequent frame.
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed before CUDA graph capture");
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize before CUDA graph capture");
        checkCuda(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "cudaStreamBeginCapture");

        if (!context->enqueueV3(stream)) {
            cudaGraph_t discardedGraph = nullptr;
            cudaStreamEndCapture(stream, &discardedGraph);
            if (discardedGraph != nullptr) {
                cudaGraphDestroy(discardedGraph);
            }
            throw std::runtime_error("TensorRT enqueueV3 failed during CUDA graph capture");
        }

        cudaGraph_t graph = nullptr;
        checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        try {
            checkCuda(
                cudaGraphInstantiate(&inferenceGraphExec, graph, 0),
                "cudaGraphInstantiate");
        } catch (...) {
            cudaGraphDestroy(graph);
            throw;
        }
        inferenceGraph = graph;
        graphInput = inputData;
    }

    ~Impl()
    {
        reset();
    }

    void reset()
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        resetInferenceGraph();
        for (auto& output : outputBuffers) {
            output.release();
        }
        outputBuffers.clear();
        views.clear();
        context.reset();
        engine.reset();
        runtime.reset();
        sessionReady = false;
        inputName.clear();
        info = ModelInputInfo();
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    void load(const std::string& modelPath, int requestedDevice)
    {
        if (sessionReady) {
            checkCuda(cudaSetDevice(device), "cudaSetDevice");
        }
        reset();
        device = requestedDevice;
        checkCuda(cudaSetDevice(device), "cudaSetDevice");

        const EngineContainer container = readEngineContainer(modelPath);
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("Unable to create TensorRT runtime");
        }
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(container.serialized.data(), container.serialized.size()));
        if (!engine) {
            throw std::runtime_error("Unable to deserialize TensorRT engine");
        }
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Unable to create TensorRT execution context");
        }

        const int tensorCount = engine->getNbIOTensors();
        int inputCount = 0;
        for (int i = 0; i < tensorCount; ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                ++inputCount;
                inputName = name;
            }
        }
        if (inputCount != 1 || inputName.empty()) {
            throw std::runtime_error("YoloTrtCuda engines must have exactly one input");
        }

        nvinfer1::Dims inputShape = engine->getTensorShape(inputName.c_str());
        const bool dynamicInput = isDynamic(inputShape);
        if (dynamicInput) {
            inputShape = engine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
            if (!context->setInputShape(inputName.c_str(), inputShape)) {
                throw std::runtime_error("Unable to set the TensorRT input profile shape");
            }
        }
        const nvinfer1::DataType inputType = engine->getTensorDataType(inputName.c_str());
        if (inputShape.nbDims != 4 || inputShape.d[0] != 1 || inputShape.d[1] <= 0 ||
            inputShape.d[2] <= 0 || inputShape.d[3] <= 0) {
            throw std::runtime_error("YoloTrtCuda requires a static batch-1 input");
        }

        if (inputType == nvinfer1::DataType::kUINT8) {
            if (!container.bakedPreprocessSet || !container.bakedPreprocess) {
                throw std::runtime_error(
                    "UINT8 input is only supported for a baked YoloTrtCuda engine");
            }
            if (inputShape.d[3] != 1 && inputShape.d[3] != 3) {
                throw std::runtime_error("Baked YoloTrtCuda input must be NHWC with 1 or 3 channels");
            }
            info.width = inputShape.d[2];
            info.height = inputShape.d[1];
            info.channels = inputShape.d[3];
            info.fp16 = false;
            info.bakedPreprocess = true;
            info.inputNHWC = true;
        } else {
            if (container.bakedPreprocessSet && container.bakedPreprocess) {
                throw std::runtime_error(
                    "Baked YoloTrtCuda engines must use a UINT8 NHWC input");
            }
            if (inputType != nvinfer1::DataType::kFLOAT &&
                inputType != nvinfer1::DataType::kHALF) {
                throw std::runtime_error("YoloTrtCuda requires a float32 or float16 input tensor");
            }
            info.width = inputShape.d[3];
            info.height = inputShape.d[2];
            info.channels = inputShape.d[1];
            info.fp16 = inputType == nvinfer1::DataType::kHALF;
            info.bakedPreprocess = false;
            info.inputNHWC = false;
        }
        info.dynamicSize = dynamicInput;

        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        outputBuffers.reserve(static_cast<size_t>(tensorCount - 1));
        bool hasYoloXOutput = false;
        int64_t yoloxDecodeOffset = 0;
        for (int i = 0; i < tensorCount; ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }

            nvinfer1::Dims shape = engine->getTensorShape(name);
            if (isDynamic(shape)) {
                shape = context->getTensorShape(name);
            }
            if (shape.nbDims != 3) {
                throw std::runtime_error("YoloTrtCuda requires a rank-3 detection output");
            }
            size_t elements = 1;
            for (int dim = 0; dim < shape.nbDims; ++dim) {
                if (shape.d[dim] <= 0) {
                    throw std::runtime_error("YoloTrtCuda output shape is not fully resolved");
                }
                elements *= static_cast<size_t>(shape.d[dim]);
            }

            const nvinfer1::DataType type = engine->getTensorDataType(name);
            if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kHALF) {
                throw std::runtime_error("YoloTrtCuda detection outputs must be float32 or float16");
            }

            OutputBuffer output;
            output.name = name;
            output.shape = toVector(shape);
            output.fp16 = type == nvinfer1::DataType::kHALF;
            const int featureCount = output.shape[1] < output.shape[2] ?
                                         static_cast<int>(output.shape[1]) :
                                         static_cast<int>(output.shape[2]);
            const int boxCount = output.shape[1] > output.shape[2] ?
                                     static_cast<int>(output.shape[1]) :
                                     static_cast<int>(output.shape[2]);
            const bool legacyYoloX = info.width == 320 && info.height == 320 &&
                                     featureCount == 7 && boxCount == 2100;
            output.yoloxDecode = container.yoloxDecodeSet ?
                                     container.yoloxDecode : legacyYoloX;
            output.objectness = output.yoloxDecode ?
                                    true :
                                    (container.objectnessSet ? container.objectness : featureCount == 85);
            output.endToEnd = container.endToEndSet && container.endToEnd;
            output.endToEndSet = container.endToEndSet;
            output.yoloxDecodeOffset = output.yoloxDecode ? yoloxDecodeOffset : 0;
            if (output.yoloxDecode) {
                yoloxDecodeOffset += static_cast<int64_t>(boxCount);
            }
            hasYoloXOutput = hasYoloXOutput || output.yoloxDecode;
            output.bytes = elements * (output.fp16 ? sizeof(uint16_t) : sizeof(float));
            checkCuda(cudaMalloc(&output.device, output.bytes), "cudaMalloc");
            try {
                checkCuda(cudaMallocHost(&output.host, output.bytes), "cudaMallocHost");
            } catch (...) {
                cudaFree(output.device);
                output.device = nullptr;
                throw;
            }
            if (!context->setTensorAddress(output.name.c_str(), output.device)) {
                throw std::runtime_error("Unable to bind a TensorRT output tensor");
            }
            outputBuffers.push_back(std::move(output));
        }
        if (outputBuffers.empty()) {
            throw std::runtime_error("YoloTrtCuda engine has no detection output");
        }
        const bool rawInput = container.rawInputSet ? container.rawInput : hasYoloXOutput;
        info.inputScale = rawInput ? 1.0f : (1.0f / 255.0f);

        views.reserve(outputBuffers.size());
        for (const auto& output : outputBuffers) {
            views.push_back({
                output.shape,
                output.fp16,
                output.objectness,
                output.yoloxDecode,
                output.endToEnd,
                output.endToEndSet,
                output.yoloxDecodeOffset,
                output.device,
                output.host,
                output.bytes});
        }
        sessionReady = true;
    }

    const std::vector<OutputView>& run(InputTensor& input)
    {
        if (!sessionReady || input.deviceData == nullptr) {
            throw std::runtime_error("YoloTrtCuda is not ready for inference");
        }
        if (graphInput != input.deviceData) {
            if (!context->setTensorAddress(inputName.c_str(), input.deviceData)) {
                throw std::runtime_error("Unable to bind a TensorRT input tensor");
            }
            captureInferenceGraph(input.deviceData);
        }

        if (inferenceGraphExec != nullptr) {
            checkCuda(cudaGraphLaunch(inferenceGraphExec, stream), "cudaGraphLaunch");
        } else {
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("TensorRT enqueueV3 failed");
            }
        }
        return views;
    }
};

InferEngine::InferEngine()
    : impl(std::make_unique<Impl>())
{
}

InferEngine::~InferEngine() = default;

bool InferEngine::loadModel(const std::string& modelPath, int device)
{
    try {
        impl->load(modelPath, device);
        return true;
    } catch (const std::exception& exception) {
        std::cerr << "[YoloTrtCuda] failed to load model: " << exception.what() << std::endl;
        impl->reset();
        return false;
    }
}

bool InferEngine::ready() const
{
    return impl->sessionReady;
}

const ModelInputInfo& InferEngine::inputInfo() const
{
    return impl->info;
}

const std::vector<OutputView>& InferEngine::run(InputTensor& input)
{
    return impl->run(input);
}

cudaStream_t InferEngine::stream() const
{
    return impl->stream;
}
