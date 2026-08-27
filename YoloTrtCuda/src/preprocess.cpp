#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define YOLOTRTCUDA_SSE 1
#if defined(_MSC_VER) || defined(__SSSE3__)
#define YOLOTRTCUDA_SSSE3 1
#endif
#endif

namespace
{
struct PixelLayout
{
    int step;
    int r;
    int g;
    int b;
};

PixelLayout pixelLayout(ImageFormat format)
{
    switch (format) {
    case ImageFormat::RGB8:
        return {3, 0, 1, 2};
    case ImageFormat::BGRA8:
        return {4, 2, 1, 0};
    case ImageFormat::RGBA8:
        return {4, 0, 1, 2};
    case ImageFormat::GRAY8:
        return {1, 0, 0, 0};
    case ImageFormat::BGR8:
    default:
        return {3, 2, 1, 0};
    }
}

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void ensureDeviceBuffer(void** pointer, size_t& capacity, size_t bytes)
{
    if (bytes <= capacity) {
        return;
    }
    if (*pointer != nullptr) {
        checkCuda(cudaFree(*pointer), "cudaFree");
        *pointer = nullptr;
        capacity = 0;
    }
    checkCuda(cudaMalloc(pointer, bytes), "cudaMalloc");
    capacity = bytes;
}

void ensureHostBuffer(void** pointer, size_t& capacity, size_t bytes)
{
    if (bytes <= capacity) {
        return;
    }
    if (*pointer != nullptr) {
        checkCuda(cudaFreeHost(*pointer), "cudaFreeHost");
        *pointer = nullptr;
        capacity = 0;
    }
    checkCuda(cudaMallocHost(pointer, bytes), "cudaMallocHost");
    capacity = bytes;
}

void buildXMap(int sourceWidth, int destinationWidth, int step, PreprocessContext& context)
{
    context.xOffset0.resize(destinationWidth);
    context.xOffset1.resize(destinationWidth);
    context.xWeight.resize(destinationWidth);
    const float ratio = static_cast<float>(sourceWidth) / static_cast<float>(destinationWidth);
    for (int x = 0; x < destinationWidth; ++x) {
        const float sourceX = (static_cast<float>(x) + 0.5f) * ratio - 0.5f;
        int x0 = static_cast<int>(std::floor(sourceX));
        float weight = sourceX - static_cast<float>(x0);
        if (x0 < 0) {
            x0 = 0;
            weight = 0.0f;
        }
        context.xOffset0[x] = x0 * step;
        context.xOffset1[x] = std::min(x0 + 1, sourceWidth - 1) * step;
        context.xWeight[x] = weight;
    }

    context.xSafe = destinationWidth;
    if (step == 3) {
        const int limit = (sourceWidth - 1) * 3;
        while (context.xSafe > 0 && context.xOffset1[context.xSafe - 1] >= limit) {
            --context.xSafe;
        }
    }
}

void convertRowRgbU8(
    const uint8_t* source,
    const PixelLayout& layout,
    int width,
    uint8_t* destination)
{
    if (layout.step == 3 && layout.r == 0) {
        std::memcpy(destination, source, static_cast<size_t>(width) * 3);
        return;
    }

    int x = 0;
#ifdef YOLOTRTCUDA_SSSE3
    if (layout.step == 3) {
        // BGR -> RGB, five pixels per iteration. The final byte is rewritten by
        // the next store or by the scalar tail.
        const __m128i swap = _mm_setr_epi8(
            2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
        for (; x + 6 <= width; x += 5) {
            const __m128i values = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(source + static_cast<size_t>(x) * 3));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(destination + static_cast<size_t>(x) * 3),
                _mm_shuffle_epi8(values, swap));
        }
    }
#endif
    for (; x < width; ++x) {
        const uint8_t* pixel = source + static_cast<size_t>(x) * layout.step;
        uint8_t* output = destination + static_cast<size_t>(x) * 3;
        output[0] = pixel[layout.r];
        output[1] = pixel[layout.g];
        output[2] = pixel[layout.b];
    }
}

void convertRowLumaU8(
    const uint8_t* source,
    const PixelLayout& layout,
    int width,
    uint8_t* destination)
{
    if (layout.step == 1) {
        std::memcpy(destination, source, static_cast<size_t>(width));
        return;
    }
    for (int x = 0; x < width; ++x, source += layout.step) {
        destination[x] = static_cast<uint8_t>(
            0.299f * source[layout.r] +
            0.587f * source[layout.g] +
            0.114f * source[layout.b] +
            0.5f);
    }
}

void hresizeRgbInterleaved(
    const uint8_t* source,
    const PixelLayout& layout,
    const PreprocessContext& context,
    int width,
    float* destination)
{
    const int* offset0 = context.xOffset0.data();
    const int* offset1 = context.xOffset1.data();
    const float* weights = context.xWeight.data();
    int x = 0;
#ifdef YOLOTRTCUDA_SSSE3
    if (layout.step == 3 || layout.step == 4) {
        const int simdEnd = layout.step == 3 ? context.xSafe : width;
        const __m128i zero = _mm_setzero_si128();
        const __m128i reorder = _mm_setr_epi8(
            static_cast<char>(layout.r),
            static_cast<char>(layout.g),
            static_cast<char>(layout.b),
            -1,
            static_cast<char>(8 + layout.r),
            static_cast<char>(8 + layout.g),
            static_cast<char>(8 + layout.b),
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1);
        for (; x < simdEnd; ++x) {
            int32_t packed0;
            int32_t packed1;
            std::memcpy(&packed0, source + offset0[x], sizeof(packed0));
            std::memcpy(&packed1, source + offset1[x], sizeof(packed1));
            const __m128i pair = _mm_unpacklo_epi64(
                _mm_cvtsi32_si128(packed0), _mm_cvtsi32_si128(packed1));
            const __m128i rgb16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(pair, reorder), zero);
            const __m128 p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgb16, zero));
            const __m128 p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgb16, zero));
            const __m128 value = _mm_add_ps(
                p0,
                _mm_mul_ps(_mm_sub_ps(p1, p0), _mm_set1_ps(weights[x])));
            _mm_storeu_ps(destination + static_cast<size_t>(x) * 3, value);
        }
    }
#endif
    for (; x < width; ++x) {
        const uint8_t* pixel0 = source + offset0[x];
        const uint8_t* pixel1 = source + offset1[x];
        const float weight = weights[x];
        float* output = destination + static_cast<size_t>(x) * 3;
        output[0] = static_cast<float>(pixel0[layout.r]) +
                    (static_cast<float>(pixel1[layout.r]) - static_cast<float>(pixel0[layout.r])) * weight;
        output[1] = static_cast<float>(pixel0[layout.g]) +
                    (static_cast<float>(pixel1[layout.g]) - static_cast<float>(pixel0[layout.g])) * weight;
        output[2] = static_cast<float>(pixel0[layout.b]) +
                    (static_cast<float>(pixel1[layout.b]) - static_cast<float>(pixel0[layout.b])) * weight;
    }
}

void hresizeLuma(
    const uint8_t* source,
    const PixelLayout& layout,
    const PreprocessContext& context,
    int width,
    float* destination)
{
    const int* offset0 = context.xOffset0.data();
    const int* offset1 = context.xOffset1.data();
    const float* weights = context.xWeight.data();
    for (int x = 0; x < width; ++x) {
        const uint8_t* pixel0 = source + offset0[x];
        const uint8_t* pixel1 = source + offset1[x];
        const float luma0 = 0.299f * pixel0[layout.r] +
                            0.587f * pixel0[layout.g] +
                            0.114f * pixel0[layout.b];
        const float luma1 = 0.299f * pixel1[layout.r] +
                            0.587f * pixel1[layout.g] +
                            0.114f * pixel1[layout.b];
        destination[x] = luma0 + (luma1 - luma0) * weights[x];
    }
}

void lerpRowToU8(
    const float* row0,
    const float* row1,
    float weight,
    int count,
    uint8_t* destination)
{
    int i = 0;
#ifdef YOLOTRTCUDA_SSE
    const __m128 w = _mm_set1_ps(weight);
    const __m128 half = _mm_set1_ps(0.5f);
    for (; i + 16 <= count; i += 16) {
        __m128i values[4];
        for (int k = 0; k < 4; ++k) {
            const __m128 a = _mm_loadu_ps(row0 + i + k * 4);
            const __m128 b = _mm_loadu_ps(row1 + i + k * 4);
            values[k] = _mm_cvttps_epi32(
                _mm_add_ps(_mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), w)), half));
        }
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(destination + i),
            _mm_packus_epi16(
                _mm_packs_epi32(values[0], values[1]),
                _mm_packs_epi32(values[2], values[3])));
    }
#endif
    for (; i < count; ++i) {
        destination[i] = static_cast<uint8_t>(
            row0[i] + (row1[i] - row0[i]) * weight + 0.5f);
    }
}
}

PreprocessResult preprocess(
    const ModelInputInfo& info,
    const ImageView& image,
    PreprocessContext& context,
    InputTensor& tensor,
    cudaStream_t stream)
{
    if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("Image data and dimensions must be valid");
    }
    if (info.width <= 0 || info.height <= 0 || (info.channels != 1 && info.channels != 3)) {
        throw std::invalid_argument("Model input dimensions or channels are unsupported");
    }

    const PixelLayout layout = pixelLayout(image.format);
    const size_t stride = image.stride != 0 ? image.stride :
        static_cast<size_t>(image.width) * layout.step;
    if (stride < static_cast<size_t>(image.width) * layout.step) {
        throw std::invalid_argument("Image stride is smaller than one image row");
    }
    if (image.channels != 0 && image.channels != layout.step) {
        throw std::invalid_argument("Image channel count does not match ImageFormat");
    }

    const float scale = std::min(
        static_cast<float>(info.width) / static_cast<float>(image.width),
        static_cast<float>(info.height) / static_cast<float>(image.height));
    const int resizedWidth = std::max(1, static_cast<int>(std::lround(image.width * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::lround(image.height * scale)));
    const int offsetX = (info.width - resizedWidth) / 2;
    const int offsetY = (info.height - resizedHeight) / 2;
    const size_t area = static_cast<size_t>(info.width) * static_cast<size_t>(info.height);
    const size_t tensorElements = area * static_cast<size_t>(info.channels);

    context.sourceWidth = image.width;
    context.sourceHeight = image.height;
    context.sourceStride = stride;

    PreprocessResult result;
    result.imageWidth = image.width;
    result.imageHeight = image.height;
    result.modelWidth = info.width;
    result.modelHeight = info.height;
    result.invScale = 1.0f / scale;
    result.padX = static_cast<float>(offsetX);
    result.padY = static_cast<float>(offsetY);

    if (info.bakedPreprocess) {
        // Baked engines accept the final uint8 NHWC image. The graph itself
        // performs the cheap cast/layout/normalization work on the GPU.
        if (tensor.sourceDevice != nullptr) {
            checkCuda(cudaFree(tensor.sourceDevice), "cudaFree");
            tensor.sourceDevice = nullptr;
            tensor.sourceBytes = 0;
        }
        ensureHostBuffer(&tensor.sourceHost, tensor.sourceHostBytes, tensorElements);
        ensureDeviceBuffer(&tensor.deviceData, tensor.deviceBytes, tensorElements);

        uint8_t* destination = static_cast<uint8_t*>(tensor.sourceHost);
        const bool padded = resizedWidth != info.width || resizedHeight != info.height;
        if (padded) {
            std::fill(destination, destination + tensorElements, static_cast<uint8_t>(114));
        }

        const uint8_t* source = static_cast<const uint8_t*>(image.data);
        if (resizedWidth == image.width && resizedHeight == image.height) {
            for (int y = 0; y < resizedHeight; ++y) {
                const uint8_t* sourceRow = source + static_cast<size_t>(y) * stride;
                uint8_t* destinationRow = destination +
                    (static_cast<size_t>(y + offsetY) * info.width + offsetX) * info.channels;
                if (info.channels == 3) {
                    convertRowRgbU8(sourceRow, layout, image.width, destinationRow);
                } else {
                    convertRowLumaU8(sourceRow, layout, image.width, destinationRow);
                }
            }
        } else {
            buildXMap(image.width, resizedWidth, layout.step, context);
            context.rowA.resize(static_cast<size_t>(resizedWidth) * info.channels + 1);
            context.rowB.resize(static_cast<size_t>(resizedWidth) * info.channels + 1);
            float* rowA = context.rowA.data();
            float* rowB = context.rowB.data();
            int rowInA = -1;
            int rowInB = -1;

            const auto horizontalResize = [&](int sourceRow, float* row) {
                const uint8_t* sourcePixels = source + static_cast<size_t>(sourceRow) * stride;
                if (info.channels == 3) {
                    hresizeRgbInterleaved(sourcePixels, layout, context, resizedWidth, row);
                } else {
                    hresizeLuma(sourcePixels, layout, context, resizedWidth, row);
                }
            };
            const auto resolveRow = [&](int sourceRow, int keepRow) -> const float* {
                if (sourceRow == rowInA) {
                    return rowA;
                }
                if (sourceRow == rowInB) {
                    return rowB;
                }
                float* target = rowInA == keepRow ? rowB : rowA;
                horizontalResize(sourceRow, target);
                (target == rowA ? rowInA : rowInB) = sourceRow;
                return target;
            };

            const float verticalRatio = static_cast<float>(image.height) /
                                        static_cast<float>(resizedHeight);
            for (int y = 0; y < resizedHeight; ++y) {
                const float sourceY = (static_cast<float>(y) + 0.5f) * verticalRatio - 0.5f;
                int y0 = static_cast<int>(std::floor(sourceY));
                float weight = sourceY - static_cast<float>(y0);
                if (y0 < 0) {
                    y0 = 0;
                    weight = 0.0f;
                }
                const int y1 = std::min(y0 + 1, image.height - 1);
                const float* sourceRow0 = resolveRow(y0, y1);
                const float* sourceRow1 = resolveRow(y1, y0);
                uint8_t* destinationRow = destination +
                    (static_cast<size_t>(y + offsetY) * info.width + offsetX) * info.channels;
                lerpRowToU8(
                    sourceRow0,
                    sourceRow1,
                    weight,
                    resizedWidth * info.channels,
                    destinationRow);
            }
        }

        tensor.shape = {1, info.height, info.width, info.channels};
        tensor.fp16 = false;
        checkCuda(
            cudaMemcpyAsync(
                tensor.deviceData,
                tensor.sourceHost,
                tensorElements,
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync");
        return result;
    }

    const size_t sourceBytes = static_cast<size_t>(image.height) * stride;
    const size_t tensorBytes = tensorElements * (info.fp16 ? sizeof(uint16_t) : sizeof(float));

    ensureHostBuffer(&tensor.sourceHost, tensor.sourceHostBytes, sourceBytes);
    ensureDeviceBuffer(&tensor.sourceDevice, tensor.sourceBytes, sourceBytes);
    ensureDeviceBuffer(&tensor.deviceData, tensor.deviceBytes, tensorBytes);

    std::memcpy(tensor.sourceHost, image.data, sourceBytes);
    checkCuda(cudaMemcpyAsync(
                  tensor.sourceDevice,
                  tensor.sourceHost,
                  sourceBytes,
                  cudaMemcpyHostToDevice,
                  stream),
              "cudaMemcpyAsync");

    tensor.shape = {1, info.channels, info.height, info.width};
    tensor.fp16 = info.fp16;
    launchLetterbox(
        tensor.sourceDevice,
        image.width,
        image.height,
        stride,
        tensor.deviceData,
        info.width,
        info.height,
        static_cast<int>(image.format),
        info.channels,
        info.fp16,
        info.inputScale,
        resizedWidth,
        resizedHeight,
        offsetX,
        offsetY,
        stream);

    return result;
}
