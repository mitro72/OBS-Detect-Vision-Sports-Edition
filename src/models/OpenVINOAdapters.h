#pragma once

#include "models/IDetectorModel.h"
#include "edgeyolo/edgeyolo_openvino.hpp"
#include "yunet/YuNetOpenVINO.h"

struct EdgeYOLOOpenVINOAdapter final : public IDetectorModel {
    edgeyolo_cpp::EdgeYOLOOpenVINO impl;

    EdgeYOLOOpenVINOAdapter(const std::string& modelPath,
                            const std::string& device,
                            int numThreads,
                            int numClasses,
                            float nmsTh,
                            float confTh)
    : impl(modelPath, device, numThreads, numClasses, nmsTh, confTh) {}

    std::vector<Object> inference(const cv::Mat& bgr) override { return impl.inference(bgr); }
    void setBBoxConfThresh(float t) override { impl.setBBoxConfThresh(t); }
};

struct YuNetOpenVINOAdapter final : public IDetectorModel {
    yunet::YuNetOpenVINO impl;

    YuNetOpenVINOAdapter(const std::string& modelPath,
                         const std::string& device,
                         int numThreads,
                         int keepTopK,
                         float nmsTh,
                         float confTh)
    : impl(modelPath, device, numThreads, keepTopK, nmsTh, confTh) {}

    std::vector<Object> inference(const cv::Mat& bgr) override { return impl.inference(bgr); }
    void setBBoxConfThresh(float t) override { impl.setBBoxConfThresh(t); }
};
