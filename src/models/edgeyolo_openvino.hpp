#pragma once

#include "models/IDetectorModel.h"
#include <openvino/openvino.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>

namespace edgeyolo_cpp {

class EdgeYOLOOpenVINO : public IDetectorModel {
public:
    EdgeYOLOOpenVINO(const std::string& modelPath,
                     const std::string& device,  // "CPU" o "GPU"
                     int numThreads,
                     int numClasses,
                     float nmsTh,
                     float confTh);

    std::vector<Object> inference(const cv::Mat& bgr) override;
    void setBBoxConfThresh(float conf) override { confTh_ = conf; }

private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest req_;

    ov::Output<const ov::Node> input_;
    ov::Output<const ov::Node> output_;

    int numClasses_ = 80;
    float nmsTh_ = 0.45f;
    float confTh_ = 0.5f;

    int inW_ = 0;
    int inH_ = 0;

    std::mutex mtx_;

    // TODO: riusa le tue funzioni (letterbox/scale) dal progetto
    // e il decoder/NMS già esistente.
};

} // namespace edgeyolo_cpp
