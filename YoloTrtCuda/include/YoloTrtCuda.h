#ifndef YOLOTRTCUDA_H
#define YOLOTRTCUDA_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#  if defined(YOLOTRTCUDA_BUILD_SHARED)
#    define YOLOTRTCUDA_API __declspec(dllexport)
#  elif defined(YOLOTRTCUDA_USE_SHARED)
#    define YOLOTRTCUDA_API __declspec(dllimport)
#  else
#    define YOLOTRTCUDA_API
#  endif
#else
#  define YOLOTRTCUDA_API __attribute__((visibility("default")))
#endif

struct DetectResultBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    int classId;
};

enum class ImageFormat
{
    BGR8,
    RGB8,
    BGRA8,
    RGBA8,
    GRAY8
};

struct ImageView
{
    const void* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    size_t stride = 0;
    ImageFormat format = ImageFormat::BGR8;
};

class YOLOTRTCUDA_API YoloTrtCuda
{
public:
    YoloTrtCuda();
    ~YoloTrtCuda();

    bool setModel(std::string modelPath);
    void setDevice(int device);
    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setImage(ImageView& image);    // the pixel data must stay valid until preprocess() returns
    void preprocess();
    void infer();
    void postprocess();
    std::vector<DetectResultBox> resultBoxes();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // YOLOTRTCUDA_H
