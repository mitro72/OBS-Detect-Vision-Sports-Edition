#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "coco_names.hpp"
#include "edgeyolo_postprocess.hpp"   // quello che ti ho fatto creare prima

namespace edgeyolo_cpp {

class EdgeYOLOOpenVINO {
public:
    EdgeYOLOOpenVINO(const std::string& modelPath,
                     const std::string& device,   // "CPU" / "GPU"
                     int numThreads,
                     int numClasses,
                     float nmsTh,
                     float confTh);

    std::vector<Object> inference(const cv::Mat& frame);
    void setBBoxConfThresh(float conf) { confTh_ = conf; }

private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest req_;

    int inW_ = 0;
    int inH_ = 0;

    int numClasses_ = 80;
    float nmsTh_ = 0.45f;
    float confTh_ = 0.5f;

    int numArray_ = 0;

    // NCHW float32 input
    std::vector<float> nchw_;

    // ORT-style letterbox (top-left, pad right/bottom with 114)
    cv::Mat static_resize_like_ort(const cv::Mat& img) const;
    void blobFromImage_like_ort(const cv::Mat& img, float* blob_data) const;
};

} // namespace edgeyolo_cpp

