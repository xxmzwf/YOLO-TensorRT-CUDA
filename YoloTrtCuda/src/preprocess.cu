#include "preprocess.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace
{
__device__ __forceinline__ float readChannel(const uint8_t* pixel, int format, int channel)
{
    if (format == 4) {
        return static_cast<float>(pixel[0]);
    }
    if (format == 0) {
        return static_cast<float>(pixel[channel == 0 ? 2 : (channel == 1 ? 1 : 0)]);
    }
    if (format == 1) {
        return static_cast<float>(pixel[channel]);
    }
    if (format == 2) {
        return static_cast<float>(pixel[channel == 0 ? 2 : (channel == 1 ? 1 : 0)]);
    }
    return static_cast<float>(pixel[channel]);
}

__device__ __forceinline__ void storeValue(void* destination, size_t index, bool outputFp16, float value)
{
    if (outputFp16) {
        static_cast<__half*>(destination)[index] = __float2half_rn(value);
    } else {
        static_cast<float*>(destination)[index] = value;
    }
}

template <bool OutputFp16>
__device__ __forceinline__ void storeFastValue(void* destination, size_t index, float value)
{
    if constexpr (OutputFp16) {
        static_cast<__half*>(destination)[index] = __float2half_rn(value);
    } else {
        static_cast<float*>(destination)[index] = value;
    }
}

__device__ void resizePixel(
    const uint8_t* source,
    int sourceWidth,
    int sourceHeight,
    size_t sourceStride,
    int format,
    int pixelX,
    int pixelY,
    int validWidth,
    int validHeight,
    int offsetX,
    int offsetY,
    float* red,
    float* green,
    float* blue)
{
    constexpr float border = 114.0f;
    *red = border;
    *green = border;
    *blue = border;

    if (pixelX < offsetX || pixelX >= offsetX + validWidth ||
        pixelY < offsetY || pixelY >= offsetY + validHeight) {
        return;
    }

    const float scaleX = static_cast<float>(sourceWidth) / static_cast<float>(validWidth);
    const float scaleY = static_cast<float>(sourceHeight) / static_cast<float>(validHeight);
    const float sourceX = (static_cast<float>(pixelX - offsetX) + 0.5f) * scaleX - 0.5f;
    const float sourceY = (static_cast<float>(pixelY - offsetY) + 0.5f) * scaleY - 0.5f;

    int x0 = static_cast<int>(floorf(sourceX));
    int y0 = static_cast<int>(floorf(sourceY));
    const float ax = sourceX - static_cast<float>(x0);
    const float ay = sourceY - static_cast<float>(y0);
    x0 = max(0, min(x0, sourceWidth - 1));
    y0 = max(0, min(y0, sourceHeight - 1));
    const int x1 = min(x0 + 1, sourceWidth - 1);
    const int y1 = min(y0 + 1, sourceHeight - 1);

    const uint8_t* p00 = source + static_cast<size_t>(y0) * sourceStride + x0 * (format == 4 ? 1 : (format == 2 || format == 3 ? 4 : 3));
    const uint8_t* p10 = source + static_cast<size_t>(y0) * sourceStride + x1 * (format == 4 ? 1 : (format == 2 || format == 3 ? 4 : 3));
    const uint8_t* p01 = source + static_cast<size_t>(y1) * sourceStride + x0 * (format == 4 ? 1 : (format == 2 || format == 3 ? 4 : 3));
    const uint8_t* p11 = source + static_cast<size_t>(y1) * sourceStride + x1 * (format == 4 ? 1 : (format == 2 || format == 3 ? 4 : 3));

    const float w00 = (1.0f - ax) * (1.0f - ay);
    const float w10 = ax * (1.0f - ay);
    const float w01 = (1.0f - ax) * ay;
    const float w11 = ax * ay;
    *red = w00 * readChannel(p00, format, 0) + w10 * readChannel(p10, format, 0) +
           w01 * readChannel(p01, format, 0) + w11 * readChannel(p11, format, 0);
    *green = w00 * readChannel(p00, format, 1) + w10 * readChannel(p10, format, 1) +
             w01 * readChannel(p01, format, 1) + w11 * readChannel(p11, format, 1);
    *blue = w00 * readChannel(p00, format, 2) + w10 * readChannel(p10, format, 2) +
            w01 * readChannel(p01, format, 2) + w11 * readChannel(p11, format, 2);
}

__global__ void letterboxKernel(
    const uint8_t* source,
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
    int offsetY)
{
    const int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const size_t area = static_cast<size_t>(destinationWidth) * destinationHeight;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int y = baseY + dy;
        if (y >= destinationHeight) {
            continue;
        }
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int x = baseX + dx;
            if (x >= destinationWidth) {
                continue;
            }

            float red;
            float green;
            float blue;
            resizePixel(
                source,
                sourceWidth,
                sourceHeight,
                sourceStride,
                sourceFormat,
                x,
                y,
                validWidth,
                validHeight,
                offsetX,
                offsetY,
                &red,
                &green,
                &blue);

            const size_t index = static_cast<size_t>(y) * destinationWidth + x;
            if (modelChannels == 1) {
                const float value = (0.299f * red + 0.587f * green + 0.114f * blue) * inputScale;
                storeValue(destination, index, outputFp16, value);
            } else {
                storeValue(destination, index, outputFp16, red * inputScale);
                storeValue(destination, area + index, outputFp16, green * inputScale);
                storeValue(destination, 2 * area + index, outputFp16, blue * inputScale);
            }
        }
    }
}

// This is the fixed BGR8/NCHW path used by the demo and by OpenCV callers.
// It follows TensorRT-YOLO's letterbox kernel closely: the transform metadata
// is computed once on the host, channel order is specialized, and read-only
// source pixels use the texture cache. The generic kernel remains available
// for the other public ImageFormat values.
template <bool OutputFp16>
__global__ void bgrLetterboxKernel(
    const uint8_t* __restrict__ source,
    int sourceWidth,
    int sourceHeight,
    size_t sourceStride,
    void* __restrict__ destination,
    int destinationWidth,
    int destinationHeight,
    float inputScale,
    int validWidth,
    int validHeight,
    int offsetX,
    int offsetY)
{
    const int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const size_t area = static_cast<size_t>(destinationWidth) * destinationHeight;
    const float scaleX = static_cast<float>(sourceWidth) * __frcp_rn(static_cast<float>(validWidth));
    const float scaleY = static_cast<float>(sourceHeight) * __frcp_rn(static_cast<float>(validHeight));
    const int validRight = offsetX + validWidth;
    const int validBottom = offsetY + validHeight;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int y = baseY + dy;
        if (y >= destinationHeight) {
            continue;
        }

#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int x = baseX + dx;
            if (x >= destinationWidth) {
                continue;
            }

            float red = 114.0f;
            float green = 114.0f;
            float blue = 114.0f;
            const bool inside = x >= offsetX && x < validRight &&
                                y >= offsetY && y < validBottom;
            if (inside) {
                const float sourceX = __fmaf_rn(
                    static_cast<float>(x - offsetX) + 0.5f, scaleX, -0.5f);
                const float sourceY = __fmaf_rn(
                    static_cast<float>(y - offsetY) + 0.5f, scaleY, -0.5f);
                const int x0 = max(0, min(__float2int_rd(sourceX), sourceWidth - 2));
                const int y0 = max(0, min(__float2int_rd(sourceY), sourceHeight - 2));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const float wx = sourceX - static_cast<float>(x0);
                const float wy = sourceY - static_cast<float>(y0);
                const float w00 = (1.0f - wx) * (1.0f - wy);
                const float w10 = wx * (1.0f - wy);
                const float w01 = (1.0f - wx) * wy;
                const float w11 = wx * wy;

                const uint8_t* p00 = source + static_cast<size_t>(y0) * sourceStride + x0 * 3;
                const uint8_t* p10 = source + static_cast<size_t>(y0) * sourceStride + x1 * 3;
                const uint8_t* p01 = source + static_cast<size_t>(y1) * sourceStride + x0 * 3;
                const uint8_t* p11 = source + static_cast<size_t>(y1) * sourceStride + x1 * 3;

                red = fmaf(w00, static_cast<float>(__ldg(p00 + 2)),
                           fmaf(w10, static_cast<float>(__ldg(p10 + 2)),
                                fmaf(w01, static_cast<float>(__ldg(p01 + 2)),
                                     w11 * static_cast<float>(__ldg(p11 + 2)))));
                green = fmaf(w00, static_cast<float>(__ldg(p00 + 1)),
                             fmaf(w10, static_cast<float>(__ldg(p10 + 1)),
                                  fmaf(w01, static_cast<float>(__ldg(p01 + 1)),
                                       w11 * static_cast<float>(__ldg(p11 + 1)))));
                blue = fmaf(w00, static_cast<float>(__ldg(p00)),
                            fmaf(w10, static_cast<float>(__ldg(p10)),
                                 fmaf(w01, static_cast<float>(__ldg(p01)),
                                      w11 * static_cast<float>(__ldg(p11)))));
            }

            const size_t index = static_cast<size_t>(y) * destinationWidth + x;
            storeFastValue<OutputFp16>(destination, index, red * inputScale);
            storeFastValue<OutputFp16>(destination, area + index, green * inputScale);
            storeFastValue<OutputFp16>(destination, 2 * area + index, blue * inputScale);
        }
    }
}
}

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
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (destinationWidth + block.x * 2 - 1) / (block.x * 2),
        (destinationHeight + block.y * 2 - 1) / (block.y * 2));

    if (sourceFormat == 0 && modelChannels == 3 && sourceWidth > 1 && sourceHeight > 1) {
        if (outputFp16) {
            bgrLetterboxKernel<true><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(source),
                sourceWidth,
                sourceHeight,
                sourceStride,
                destination,
                destinationWidth,
                destinationHeight,
                inputScale,
                validWidth,
                validHeight,
                offsetX,
                offsetY);
        } else {
            bgrLetterboxKernel<false><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(source),
                sourceWidth,
                sourceHeight,
                sourceStride,
                destination,
                destinationWidth,
                destinationHeight,
                inputScale,
                validWidth,
                validHeight,
                offsetX,
                offsetY);
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("bgrLetterboxKernel: ") + cudaGetErrorString(error));
        }
        return;
    }

    letterboxKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(source),
        sourceWidth,
        sourceHeight,
        sourceStride,
        destination,
        destinationWidth,
        destinationHeight,
        sourceFormat,
        modelChannels,
        outputFp16,
        inputScale,
        validWidth,
        validHeight,
        offsetX,
        offsetY);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("letterboxKernel: ") + cudaGetErrorString(error));
    }
}
