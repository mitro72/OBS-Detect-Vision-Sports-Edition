#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

struct Object; // già definito nel tuo progetto

class IDetectorModel {
public:
    virtual ~IDetectorModel() = default;
    virtual std::vector<Object> inference(const cv::Mat& bgr) = 0;
    virtual void setBBoxConfThresh(float conf) = 0;
};
