#include "postprocess.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
struct Candidate
{
    float x;
    float y;
    float width;
    float height;
    float score;
    int classId;
};

template <typename T>
__device__ __forceinline__ float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<__half>(__half value)
{
    return __half2float(value);
}

__device__ __forceinline__ float clampValue(float value, float low, float high)
{
    return value < low ? low : (value > high ? high : value);
}

__device__ bool decodeYoloXBox(
    int index,
    int64_t indexOffset,
    int modelWidth,
    int modelHeight,
    float& centerX,
    float& centerY,
    float& width,
    float& height)
{
    const int64_t globalIndex = indexOffset + static_cast<int64_t>(index);
    int64_t offset = 0;
    int gridWidth = 0;
    int gridHeight = 0;
    int gridX = 0;
    int gridY = 0;
    int stride = 0;
    for (int candidateStride = 8; candidateStride <= 64; candidateStride *= 2) {
        const int levelWidth = modelWidth / candidateStride;
        const int levelHeight = modelHeight / candidateStride;
        const int64_t levelCount = static_cast<int64_t>(levelWidth) * levelHeight;
        if (levelCount <= 0) {
            continue;
        }
        if (globalIndex < offset + levelCount) {
            const int localIndex = static_cast<int>(globalIndex - offset);
            gridWidth = levelWidth;
            gridHeight = levelHeight;
            gridX = localIndex % gridWidth;
            gridY = localIndex / gridWidth;
            stride = candidateStride;
            break;
        }
        offset += levelCount;
    }

    if (stride == 0 || gridWidth <= 0 || gridHeight <= 0) {
        return false;
    }

    centerX = (centerX + static_cast<float>(gridX)) * static_cast<float>(stride);
    centerY = (centerY + static_cast<float>(gridY)) * static_cast<float>(stride);
    width = expf(clampValue(width, -20.0f, 20.0f)) * static_cast<float>(stride);
    height = expf(clampValue(height, -20.0f, 20.0f)) * static_cast<float>(stride);
    return true;
}

__device__ void storeCenterCandidate(
    Candidate* candidates,
    int* candidateCount,
    int capacity,
    float centerX,
    float centerY,
    float width,
    float height,
    float score,
    int classId,
    float padX,
    float padY,
    float invScale,
    int imageWidth,
    int imageHeight)
{
    if (!isfinite(score) || score <= 0.0f) {
        return;
    }
    float left = (centerX - 0.5f * width - padX) * invScale;
    float top = (centerY - 0.5f * height - padY) * invScale;
    float right = (centerX + 0.5f * width - padX) * invScale;
    float bottom = (centerY + 0.5f * height - padY) * invScale;
    left = clampValue(left, 0.0f, static_cast<float>(imageWidth));
    top = clampValue(top, 0.0f, static_cast<float>(imageHeight));
    right = clampValue(right, 0.0f, static_cast<float>(imageWidth));
    bottom = clampValue(bottom, 0.0f, static_cast<float>(imageHeight));
    if (right <= left || bottom <= top) {
        return;
    }

    const int index = atomicAdd(candidateCount, 1);
    if (index >= capacity) {
        return;
    }
    candidates[index] = {left, top, right - left, bottom - top, score, classId};
}

__device__ void storeBoxCandidate(
    Candidate* candidates,
    int* candidateCount,
    int capacity,
    float left,
    float top,
    float right,
    float bottom,
    float score,
    int classId,
    float padX,
    float padY,
    float invScale,
    int imageWidth,
    int imageHeight)
{
    if (!isfinite(score) || score <= 0.0f) {
        return;
    }
    left = clampValue((left - padX) * invScale, 0.0f, static_cast<float>(imageWidth));
    top = clampValue((top - padY) * invScale, 0.0f, static_cast<float>(imageHeight));
    right = clampValue((right - padX) * invScale, 0.0f, static_cast<float>(imageWidth));
    bottom = clampValue((bottom - padY) * invScale, 0.0f, static_cast<float>(imageHeight));
    if (right <= left || bottom <= top) {
        return;
    }

    const int index = atomicAdd(candidateCount, 1);
    if (index >= capacity) {
        return;
    }
    candidates[index] = {left, top, right - left, bottom - top, score, classId};
}

template <typename T>
__global__ void collectRowMajor(
    const T* values,
    int boxCount,
    int features,
    int classes,
    bool objectness,
    bool yoloxDecode,
    int64_t yoloxDecodeOffset,
    int modelWidth,
    int modelHeight,
    float confidence,
    float padX,
    float padY,
    float invScale,
    int imageWidth,
    int imageHeight,
    Candidate* candidates,
    int* candidateCount)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boxCount) {
        return;
    }

    const T* row = values + static_cast<size_t>(index) * features;
    const int classOffset = objectness ? 5 : 4;
    const float objectnessScore = objectness ? toFloat(row[4]) : 1.0f;
    if (objectness && objectnessScore <= confidence) {
        return;
    }

    float bestScore = toFloat(row[classOffset]);
    int classId = 0;
    for (int c = 1; c < classes; ++c) {
        const float score = toFloat(row[classOffset + c]);
        if (score > bestScore) {
            bestScore = score;
            classId = c;
        }
    }
    const float score = objectnessScore * bestScore;
    if (score <= confidence) {
        return;
    }
    float centerX = toFloat(row[0]);
    float centerY = toFloat(row[1]);
    float width = toFloat(row[2]);
    float height = toFloat(row[3]);
    if (yoloxDecode && !decodeYoloXBox(
                           index, yoloxDecodeOffset, modelWidth, modelHeight,
                           centerX, centerY, width, height)) {
        return;
    }
    storeCenterCandidate(
        candidates,
        candidateCount,
        boxCount,
        centerX,
        centerY,
        width,
        height,
        score,
        classId,
        padX,
        padY,
        invScale,
        imageWidth,
        imageHeight);
}

template <typename T>
__global__ void collectPlanar(
    const T* values,
    int boxCount,
    int features,
    int classes,
    bool objectness,
    bool yoloxDecode,
    int64_t yoloxDecodeOffset,
    int modelWidth,
    int modelHeight,
    float confidence,
    float padX,
    float padY,
    float invScale,
    int imageWidth,
    int imageHeight,
    Candidate* candidates,
    int* candidateCount)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boxCount) {
        return;
    }

    const int classOffset = objectness ? 5 : 4;
    const T* classData = values + static_cast<size_t>(classOffset) * boxCount;
    float bestScore = toFloat(classData[index]);
    int classId = 0;
    for (int c = 1; c < classes; ++c) {
        const float score = toFloat(classData[static_cast<size_t>(c) * boxCount + index]);
        if (score > bestScore) {
            bestScore = score;
            classId = c;
        }
    }
    const float objectnessScore = objectness ? toFloat(values[4 * boxCount + index]) : 1.0f;
    if (objectness && objectnessScore <= confidence) {
        return;
    }

    float centerX = toFloat(values[index]);
    float centerY = toFloat(values[boxCount + index]);
    float width = toFloat(values[2 * boxCount + index]);
    float height = toFloat(values[3 * boxCount + index]);
    if (yoloxDecode && !decodeYoloXBox(
                           index, yoloxDecodeOffset, modelWidth, modelHeight,
                           centerX, centerY, width, height)) {
        return;
    }
    storeCenterCandidate(
        candidates,
        candidateCount,
        boxCount,
        centerX,
        centerY,
        width,
        height,
        objectnessScore * bestScore,
        classId,
        padX,
        padY,
        invScale,
        imageWidth,
        imageHeight);
}

template <typename T>
__global__ void collectEndToEnd(
    const T* values,
    int boxCount,
    float confidence,
    float padX,
    float padY,
    float invScale,
    int imageWidth,
    int imageHeight,
    Candidate* candidates,
    int* candidateCount)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= boxCount) {
        return;
    }
    const T* row = values + static_cast<size_t>(index) * 6;
    const float score = toFloat(row[4]);
    if (score <= confidence) {
        return;
    }
    storeBoxCandidate(
        candidates,
        candidateCount,
        boxCount,
        toFloat(row[0]),
        toFloat(row[1]),
        toFloat(row[2]),
        toFloat(row[3]),
        score,
        static_cast<int>(toFloat(row[5])),
        padX,
        padY,
        invScale,
        imageWidth,
        imageHeight);
}

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void ensureBuffers(PostprocessContext& context, int capacity)
{
    if (capacity <= context.capacity) {
        return;
    }
    context.release();
    checkCuda(cudaMalloc(&context.candidates, static_cast<size_t>(capacity) * sizeof(Candidate)), "cudaMalloc");
    checkCuda(cudaMalloc(&context.candidateCount, sizeof(int)), "cudaMalloc");
    context.capacity = capacity;
}
}

void launchCandidateCollection(
    const OutputView& output,
    const PreprocessResult& preprocess,
    float confidenceThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& candidates,
    cudaStream_t stream)
{
    if (output.shape.size() != 3 || output.shape[0] != 1 || output.shape[1] <= 0 || output.shape[2] <= 0) {
        throw std::runtime_error("YoloTrtCuda detection output must have shape [1, N, features] or [1, features, N]");
    }

    const int rows = static_cast<int>(output.shape[1]);
    const int cols = static_cast<int>(output.shape[2]);
    const bool legacyEndToEnd = !output.endToEndSet && !output.objectness && cols == 6;
    const bool endToEnd = output.endToEnd || legacyEndToEnd;
    const bool rowMajor = endToEnd || rows > cols;
    const int boxCount = rowMajor ? rows : cols;
    const int features = rowMajor ? cols : rows;
    if (boxCount <= 0 || features <= 4) {
        candidates.clear();
        return;
    }

    ensureBuffers(context, boxCount);
    checkCuda(cudaMemsetAsync(context.candidateCount, 0, sizeof(int), stream), "cudaMemsetAsync");
    const dim3 block(256);
    const dim3 grid((boxCount + block.x - 1) / block.x);
    const float padX = preprocess.padX;
    const float padY = preprocess.padY;
    const float invScale = preprocess.invScale;
    const bool objectness = output.objectness;
    const bool yoloxDecode = output.yoloxDecode;

    if (output.fp16) {
        const auto* values = static_cast<const __half*>(output.data);
        if (endToEnd) {
            collectEndToEnd<<<grid, block, 0, stream>>>(
                values, boxCount, confidenceThreshold, padX, padY, invScale,
                preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        } else if (rowMajor) {
            const int classes = features - (objectness ? 5 : 4);
            if (classes <= 0) {
                throw std::runtime_error("YoloTrtCuda row-major output has no class scores");
            }
            collectRowMajor<<<grid, block, 0, stream>>>(
                values, boxCount, features, classes, objectness, yoloxDecode,
                output.yoloxDecodeOffset,
                preprocess.modelWidth, preprocess.modelHeight, confidenceThreshold,
                padX, padY, invScale, preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        } else {
            const int classes = features - (objectness ? 5 : 4);
            if (classes <= 0) {
                throw std::runtime_error("YoloTrtCuda planar output has no class scores");
            }
            collectPlanar<<<grid, block, 0, stream>>>(
                values, boxCount, features, classes, objectness, yoloxDecode,
                output.yoloxDecodeOffset,
                preprocess.modelWidth, preprocess.modelHeight, confidenceThreshold,
                padX, padY, invScale, preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        }
    } else {
        const auto* values = static_cast<const float*>(output.data);
        if (endToEnd) {
            collectEndToEnd<<<grid, block, 0, stream>>>(
                values, boxCount, confidenceThreshold, padX, padY, invScale,
                preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        } else if (rowMajor) {
            const int classes = features - (objectness ? 5 : 4);
            if (classes <= 0) {
                throw std::runtime_error("YoloTrtCuda row-major output has no class scores");
            }
            collectRowMajor<<<grid, block, 0, stream>>>(
                values, boxCount, features, classes, objectness, yoloxDecode,
                output.yoloxDecodeOffset,
                preprocess.modelWidth, preprocess.modelHeight, confidenceThreshold,
                padX, padY, invScale, preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        } else {
            const int classes = features - (objectness ? 5 : 4);
            if (classes <= 0) {
                throw std::runtime_error("YoloTrtCuda planar output has no class scores");
            }
            collectPlanar<<<grid, block, 0, stream>>>(
                values, boxCount, features, classes, objectness, yoloxDecode,
                output.yoloxDecodeOffset,
                preprocess.modelWidth, preprocess.modelHeight, confidenceThreshold,
                padX, padY, invScale, preprocess.imageWidth, preprocess.imageHeight,
                static_cast<Candidate*>(context.candidates), context.candidateCount);
        }
    }
    checkCuda(cudaGetLastError(), "YoloTrtCuda postprocess kernel");

    int candidateCount = 0;
    checkCuda(cudaMemcpyAsync(
                  &candidateCount,
                  context.candidateCount,
                  sizeof(candidateCount),
                  cudaMemcpyDeviceToHost,
                  stream),
              "cudaMemcpyAsync");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    candidateCount = std::min(candidateCount, boxCount);
    candidates.resize(static_cast<size_t>(candidateCount));
    if (candidateCount > 0) {
        checkCuda(cudaMemcpy(
                      candidates.data(),
                      context.candidates,
                      static_cast<size_t>(candidateCount) * sizeof(Candidate),
                      cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    }
}
