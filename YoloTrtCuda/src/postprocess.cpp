#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <stdexcept>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define YOLOTRTCUDA_SSE 1
#endif

PostprocessContext::~PostprocessContext()
{
    release();
}

void PostprocessContext::release()
{
    if (candidates != nullptr) {
        cudaFree(candidates);
        candidates = nullptr;
    }
    if (candidateCount != nullptr) {
        cudaFree(candidateCount);
        candidateCount = nullptr;
    }
    capacity = 0;
    candidateResults.clear();
    bestScores.clear();
    bestHalfScores.clear();
    nmsOrder.clear();
    nmsSuppressed.clear();
}

namespace
{
void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

float clampValue(float value, float low, float high)
{
    return value < low ? low : (value > high ? high : value);
}

uint16_t floatToHalfBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    bits &= 0x7FFFFFFFu;
    if (bits >= 0x477FF000u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (bits < 0x38800000u) {
        return static_cast<uint16_t>(sign);
    }
    bits += 0x00001000u;
    return static_cast<uint16_t>(sign | ((bits - 0x38000000u) >> 13));
}

float halfBitsToFloat(uint16_t half)
{
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t magnitude = half & 0x7FFFu;
    const uint32_t bits = magnitude >= 0x0400u ? sign | ((magnitude + 0x1C000u) << 13) : sign;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void addBox(
    std::vector<DetectResultBox>& boxes,
    const PreprocessResult& preprocess,
    float left,
    float top,
    float right,
    float bottom,
    float score,
    int classId)
{
    left = clampValue(
        (left - preprocess.padX) * preprocess.invScale,
        0.0f,
        static_cast<float>(preprocess.imageWidth));
    top = clampValue(
        (top - preprocess.padY) * preprocess.invScale,
        0.0f,
        static_cast<float>(preprocess.imageHeight));
    right = clampValue(
        (right - preprocess.padX) * preprocess.invScale,
        0.0f,
        static_cast<float>(preprocess.imageWidth));
    bottom = clampValue(
        (bottom - preprocess.padY) * preprocess.invScale,
        0.0f,
        static_cast<float>(preprocess.imageHeight));
    const float width = right - left;
    const float height = bottom - top;
    if (width <= 0.0f || height <= 0.0f || !std::isfinite(score)) {
        return;
    }
    boxes.push_back({left, top, width, height, score, classId});
}

void addCenterBox(
    std::vector<DetectResultBox>& boxes,
    const PreprocessResult& preprocess,
    float centerX,
    float centerY,
    float width,
    float height,
    float score,
    int classId)
{
    addBox(
        boxes,
        preprocess,
        centerX - 0.5f * width,
        centerY - 0.5f * height,
        centerX + 0.5f * width,
        centerY + 0.5f * height,
        score,
        classId);
}

bool decodeYoloXBox(
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
    width = std::exp(clampValue(width, -20.0f, 20.0f)) * static_cast<float>(stride);
    height = std::exp(clampValue(height, -20.0f, 20.0f)) * static_cast<float>(stride);
    return true;
}

float maxScoreFloat(const float* scores, int count)
{
    int i = 0;
    float best = scores[0];
#ifdef YOLOTRTCUDA_SSE
    if (count >= 8) {
        __m128 vectorBest = _mm_loadu_ps(scores);
        for (i = 4; i + 4 <= count; i += 4) {
            vectorBest = _mm_max_ps(vectorBest, _mm_loadu_ps(scores + i));
        }
        vectorBest = _mm_max_ps(vectorBest, _mm_movehl_ps(vectorBest, vectorBest));
        vectorBest = _mm_max_ps(vectorBest, _mm_shuffle_ps(vectorBest, vectorBest, 0x55));
        best = _mm_cvtss_f32(vectorBest);
    }
#endif
    for (; i < count; ++i) {
        best = std::max(best, scores[i]);
    }
    return best;
}

int findClassFloat(const float* scores, int count, float value)
{
    int classId = 0;
    while (classId + 1 < count && scores[classId] != value) {
        ++classId;
    }
    return classId;
}

uint16_t maxScoreHalf(const uint16_t* scores, int count)
{
    int i = 0;
    uint16_t best = scores[0];
#ifdef YOLOTRTCUDA_SSE
    if (count >= 16) {
        __m128i vectorBest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scores));
        for (i = 8; i + 8 <= count; i += 8) {
            vectorBest = _mm_max_epi16(
                vectorBest,
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(scores + i)));
        }
        vectorBest = _mm_max_epi16(vectorBest, _mm_srli_si128(vectorBest, 8));
        vectorBest = _mm_max_epi16(vectorBest, _mm_srli_si128(vectorBest, 4));
        vectorBest = _mm_max_epi16(vectorBest, _mm_srli_si128(vectorBest, 2));
        best = static_cast<uint16_t>(_mm_extract_epi16(vectorBest, 0));
    }
#endif
    for (; i < count; ++i) {
        best = std::max(best, scores[i]);
    }
    return best;
}

int findClassHalf(const uint16_t* scores, int count, uint16_t value)
{
    int classId = 0;
    while (classId + 1 < count && scores[classId] != value) {
        ++classId;
    }
    return classId;
}

void collectRowMajor(
    const OutputView& output,
    const PreprocessResult& preprocess,
    float confidence,
    std::vector<DetectResultBox>& boxes)
{
    const int boxCount = static_cast<int>(output.shape[1]);
    const int features = static_cast<int>(output.shape[2]);
    const int classOffset = output.objectness ? 5 : 4;
    const int classes = features - classOffset;
    if (classes <= 0) {
        return;
    }

    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        for (int index = 0; index < boxCount; ++index) {
            const float* row = values + static_cast<size_t>(index) * features;
            const float objectness = output.objectness ? row[4] : 1.0f;
            if (output.objectness && objectness <= confidence) {
                continue;
            }
            const float* scores = row + classOffset;
            const float bestScore = maxScoreFloat(scores, classes);
            const float score = objectness * bestScore;
            if (score <= confidence) {
                continue;
            }
            const int classId = findClassFloat(scores, classes, bestScore);
            float centerX = row[0];
            float centerY = row[1];
            float width = row[2];
            float height = row[3];
            if (output.yoloxDecode && !decodeYoloXBox(
                                           index,
                                           output.yoloxDecodeOffset,
                                           preprocess.modelWidth,
                                           preprocess.modelHeight,
                                           centerX,
                                           centerY,
                                           width,
                                           height)) {
                continue;
            }
            addCenterBox(
                boxes,
                preprocess,
                centerX,
                centerY,
                width,
                height,
                score,
                classId);
        }
        return;
    }

    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t confidenceBits = floatToHalfBits(confidence);
    for (int index = 0; index < boxCount; ++index) {
        const uint16_t* row = values + static_cast<size_t>(index) * features;
        if (output.objectness && row[4] <= confidenceBits) {
            continue;
        }
        const uint16_t* scores = row + classOffset;
        const uint16_t bestBits = maxScoreHalf(scores, classes);
        float score = halfBitsToFloat(bestBits);
        if (output.objectness) {
            score *= halfBitsToFloat(row[4]);
        }
        if (score <= confidence) {
            continue;
        }
        const int classId = findClassHalf(scores, classes, bestBits);
        float centerX = halfBitsToFloat(row[0]);
        float centerY = halfBitsToFloat(row[1]);
        float width = halfBitsToFloat(row[2]);
        float height = halfBitsToFloat(row[3]);
        if (output.yoloxDecode && !decodeYoloXBox(
                                       index,
                                       output.yoloxDecodeOffset,
                                       preprocess.modelWidth,
                                       preprocess.modelHeight,
                                       centerX,
                                       centerY,
                                       width,
                                       height)) {
            continue;
        }
        addCenterBox(
            boxes,
            preprocess,
            centerX,
            centerY,
            width,
            height,
            score,
            classId);
    }
}

void planarMaxFloat(
    const float* classData,
    int classes,
    size_t count,
    std::vector<float>& best)
{
    best.assign(classData, classData + count);
    float* destination = best.data();
    for (int classId = 1; classId < classes; ++classId) {
        const float* scores = classData + static_cast<size_t>(classId) * count;
        size_t index = 0;
#ifdef YOLOTRTCUDA_SSE
        for (; index + 4 <= count; index += 4) {
            _mm_storeu_ps(
                destination + index,
                _mm_max_ps(
                    _mm_loadu_ps(destination + index),
                    _mm_loadu_ps(scores + index)));
        }
#endif
        for (; index < count; ++index) {
            destination[index] = std::max(destination[index], scores[index]);
        }
    }
}

void planarMaxHalf(
    const uint16_t* classData,
    int classes,
    size_t count,
    std::vector<uint16_t>& best)
{
    best.assign(classData, classData + count);
    uint16_t* destination = best.data();
    for (int classId = 1; classId < classes; ++classId) {
        const uint16_t* scores = classData + static_cast<size_t>(classId) * count;
        size_t index = 0;
#ifdef YOLOTRTCUDA_SSE
        for (; index + 8 <= count; index += 8) {
            const __m128i bestBits = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(destination + index));
            const __m128i scoreBits = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(scores + index));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(destination + index),
                _mm_max_epi16(bestBits, scoreBits));
        }
#endif
        for (; index < count; ++index) {
            destination[index] = std::max(destination[index], scores[index]);
        }
    }
}

void collectPlanar(
    const OutputView& output,
    const PreprocessResult& preprocess,
    float confidence,
    PostprocessContext& context)
{
    const int boxCount = static_cast<int>(output.shape[2]);
    const int features = static_cast<int>(output.shape[1]);
    const int classOffset = output.objectness ? 5 : 4;
    const int classes = features - classOffset;
    if (classes <= 0) {
        return;
    }

    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        const float* xData = values;
        const float* yData = values + boxCount;
        const float* widthData = values + 2 * boxCount;
        const float* heightData = values + 3 * boxCount;
        const float* classData = values + classOffset * boxCount;
        planarMaxFloat(classData, classes, static_cast<size_t>(boxCount), context.bestScores);
        for (int index = 0; index < boxCount; ++index) {
            const float bestScore = context.bestScores[static_cast<size_t>(index)];
            const float objectness = output.objectness ? values[4 * boxCount + index] : 1.0f;
            if ((output.objectness && objectness <= confidence) ||
                objectness * bestScore <= confidence) {
                continue;
            }
            int classId = 0;
            while (classId + 1 < classes &&
                   classData[static_cast<size_t>(classId) * boxCount + index] != bestScore) {
                ++classId;
            }
            float centerX = xData[index];
            float centerY = yData[index];
            float width = widthData[index];
            float height = heightData[index];
            if (output.yoloxDecode && !decodeYoloXBox(
                                           index,
                                           output.yoloxDecodeOffset,
                                           preprocess.modelWidth,
                                           preprocess.modelHeight,
                                           centerX,
                                           centerY,
                                           width,
                                           height)) {
                continue;
            }
            addCenterBox(
                context.candidateResults,
                preprocess,
                centerX,
                centerY,
                width,
                height,
                objectness * bestScore,
                classId);
        }
        return;
    }

    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t* xData = values;
    const uint16_t* yData = values + boxCount;
    const uint16_t* widthData = values + 2 * boxCount;
    const uint16_t* heightData = values + 3 * boxCount;
    const uint16_t* classData = values + classOffset * boxCount;
    const uint16_t confidenceBits = floatToHalfBits(confidence);
    planarMaxHalf(classData, classes, static_cast<size_t>(boxCount), context.bestHalfScores);
    for (int index = 0; index < boxCount; ++index) {
        const uint16_t objectnessBits = output.objectness ? values[4 * boxCount + index] :
                                                               floatToHalfBits(1.0f);
        if (output.objectness && objectnessBits <= confidenceBits) {
            continue;
        }
        const uint16_t bestBits = context.bestHalfScores[static_cast<size_t>(index)];
        const float objectness = halfBitsToFloat(objectnessBits);
        const float score = objectness * halfBitsToFloat(bestBits);
        if (score <= confidence) {
            continue;
        }
        int classId = 0;
        while (classId + 1 < classes &&
               classData[static_cast<size_t>(classId) * boxCount + index] != bestBits) {
            ++classId;
        }
        float centerX = halfBitsToFloat(xData[index]);
        float centerY = halfBitsToFloat(yData[index]);
        float width = halfBitsToFloat(widthData[index]);
        float height = halfBitsToFloat(heightData[index]);
        if (output.yoloxDecode && !decodeYoloXBox(
                                       index,
                                       output.yoloxDecodeOffset,
                                       preprocess.modelWidth,
                                       preprocess.modelHeight,
                                       centerX,
                                       centerY,
                                       width,
                                       height)) {
            continue;
        }
        addCenterBox(
            context.candidateResults,
            preprocess,
            centerX,
            centerY,
            width,
            height,
            score,
            classId);
    }
}

void collectEndToEnd(
    const OutputView& output,
    const PreprocessResult& preprocess,
    float confidence,
    std::vector<DetectResultBox>& boxes)
{
    const int boxCount = static_cast<int>(output.shape[1]);
    if (!output.fp16) {
        const float* values = static_cast<const float*>(output.data);
        for (int index = 0; index < boxCount; ++index) {
            const float* row = values + static_cast<size_t>(index) * 6;
            if (row[4] <= confidence) {
                continue;
            }
            addBox(
                boxes,
                preprocess,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                static_cast<int>(row[5]));
        }
        return;
    }

    const uint16_t* values = static_cast<const uint16_t*>(output.data);
    const uint16_t confidenceBits = floatToHalfBits(confidence);
    for (int index = 0; index < boxCount; ++index) {
        const uint16_t* row = values + static_cast<size_t>(index) * 6;
        if (row[4] <= confidenceBits) {
            continue;
        }
        addBox(
            boxes,
            preprocess,
            halfBitsToFloat(row[0]),
            halfBitsToFloat(row[1]),
            halfBitsToFloat(row[2]),
            halfBitsToFloat(row[3]),
            halfBitsToFloat(row[4]),
            static_cast<int>(halfBitsToFloat(row[5])));
    }
}

float intersectionOverUnion(const DetectResultBox& lhs, const DetectResultBox& rhs)
{
    const float left = std::max(lhs.x, rhs.x);
    const float top = std::max(lhs.y, rhs.y);
    const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const float width = right - left;
    const float height = bottom - top;
    if (width <= 0.0f || height <= 0.0f) {
        return 0.0f;
    }
    const float intersection = width * height;
    return intersection / (lhs.width * lhs.height + rhs.width * rhs.height - intersection);
}

void applyNms(
    PostprocessContext& context,
    float threshold,
    std::vector<DetectResultBox>& results)
{
    const std::vector<DetectResultBox>& candidates = context.candidateResults;
    if (candidates.empty()) {
        return;
    }

    context.nmsOrder.resize(candidates.size());
    std::iota(context.nmsOrder.begin(), context.nmsOrder.end(), 0);
    std::sort(context.nmsOrder.begin(), context.nmsOrder.end(), [&candidates](int lhs, int rhs) {
        return candidates[lhs].score > candidates[rhs].score;
    });
    context.nmsSuppressed.resize(candidates.size());
    std::fill(context.nmsSuppressed.begin(), context.nmsSuppressed.end(), 0);

    results.reserve(candidates.size());
    for (size_t i = 0; i < context.nmsOrder.size(); ++i) {
        const int current = context.nmsOrder[i];
        if (context.nmsSuppressed[current]) {
            continue;
        }
        const DetectResultBox& keep = candidates[current];
        results.push_back(keep);
        for (size_t j = i + 1; j < context.nmsOrder.size(); ++j) {
            const int other = context.nmsOrder[j];
            if (!context.nmsSuppressed[other] &&
                candidates[other].classId == keep.classId &&
                intersectionOverUnion(keep, candidates[other]) > threshold) {
                context.nmsSuppressed[other] = 1;
            }
        }
    }
}
}

void postprocess(
    const std::vector<OutputView>& outputs,
    const PreprocessResult& preprocessResult,
    float confidenceThreshold,
    float nmsThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& results,
    cudaStream_t stream)
{
    results.clear();
    context.candidateResults.clear();
    if (outputs.empty()) {
        return;
    }
    if (preprocessResult.imageWidth <= 0 || preprocessResult.imageHeight <= 0) {
        return;
    }
    if (confidenceThreshold < 0.0f || confidenceThreshold > 1.0f ||
        nmsThreshold < 0.0f || nmsThreshold > 1.0f) {
        throw std::invalid_argument("Confidence and NMS thresholds must be in [0, 1]");
    }

    std::vector<OutputView> hostOutputs;
    hostOutputs.reserve(outputs.size());
    for (const OutputView& deviceOutput : outputs) {
        if (deviceOutput.data == nullptr || deviceOutput.hostData == nullptr ||
            deviceOutput.bytes == 0) {
            continue;
        }
        checkCuda(
            cudaMemcpyAsync(
                deviceOutput.hostData,
                deviceOutput.data,
                deviceOutput.bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync output");
        OutputView hostOutput = deviceOutput;
        hostOutput.data = deviceOutput.hostData;
        hostOutputs.push_back(hostOutput);
    }
    if (hostOutputs.empty()) {
        return;
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize output");

    for (const OutputView& output : hostOutputs) {
        if (output.shape.size() != 3 || output.shape[0] != 1 ||
            output.shape[1] <= 0 || output.shape[2] <= 0) {
            continue;
        }

        const int64_t rows = output.shape[1];
        const int64_t columns = output.shape[2];
        const bool legacyEndToEnd = !output.endToEndSet && !output.objectness && columns == 6;
        const bool endToEnd = output.endToEnd || legacyEndToEnd;
        if (endToEnd) {
            if (columns == 6) {
                collectEndToEnd(
                    output,
                    preprocessResult,
                    confidenceThreshold,
                    context.candidateResults);
            }
            continue;
        }

        const bool rowMajor = rows > columns;
        const int features = static_cast<int>(rowMajor ? columns : rows);
        if (features <= 4) {
            continue;
        }
        if (rowMajor) {
            collectRowMajor(
                output,
                preprocessResult,
                confidenceThreshold,
                context.candidateResults);
        } else {
            collectPlanar(output, preprocessResult, confidenceThreshold, context);
        }
    }
    applyNms(context, nmsThreshold, results);
}
