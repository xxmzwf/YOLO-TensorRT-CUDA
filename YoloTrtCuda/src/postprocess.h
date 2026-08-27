#ifndef YOLOTRTCUDA_POSTPROCESS_H
#define YOLOTRTCUDA_POSTPROCESS_H

#include <cuda_runtime_api.h>

#include <vector>

#include "YoloTrtCuda.h"
#include "infer.h"
#include "preprocess.h"

struct PostprocessContext
{
    void* candidates = nullptr;
    int* candidateCount = nullptr;
    int capacity = 0;
    std::vector<DetectResultBox> candidateResults;
    std::vector<float> bestScores;
    std::vector<uint16_t> bestHalfScores;
    std::vector<int> nmsOrder;
    std::vector<unsigned char> nmsSuppressed;

    PostprocessContext() = default;
    PostprocessContext(const PostprocessContext&) = delete;
    PostprocessContext& operator=(const PostprocessContext&) = delete;
    ~PostprocessContext();

    void release();
};

void postprocess(
    const std::vector<OutputView>& outputs,
    const PreprocessResult& preprocess,
    float confidenceThreshold,
    float nmsThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& results,
    cudaStream_t stream);

void launchCandidateCollection(
    const OutputView& output,
    const PreprocessResult& preprocess,
    float confidenceThreshold,
    PostprocessContext& context,
    std::vector<DetectResultBox>& candidates,
    cudaStream_t stream);

#endif // YOLOTRTCUDA_POSTPROCESS_H
