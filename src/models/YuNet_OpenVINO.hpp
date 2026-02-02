#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <vector>
#include <array>
#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>

#include "ort-model/types.hpp"   // Object
// Se vuoi riusare gli helper NMS/sort senza ORT, spostali in utils comuni.
// Per ora li copiamo qui (vedi cpp) oppure includiamo "ort-model/utils.hpp" se contiene Object helpers.

namespace yunet {

class YuNetOpenVINO {
public:
    YuNetOpenVINO(const std::string& modelPath,
                  const std::string& device,   // "CPU"/"GPU"
                  int numThreads,
                  int keep_topk = 50,
                  float nms_th = 0.45f,
                  float conf_th = 0.3f);

    std::vector<Object> inference(const cv::Mat &frame);
    void setBBoxConfThresh(float t) { bbox_conf_thresh_ = t; }
    void setNmsThresh(float t) { nms_thresh_ = t; }

private:
    // OpenVINO
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest req_;

    // Params (identici a YuNetONNX)
    int keep_topk_ = 50;
    int divisor_ = 32;
    int padH_ = 0;
    int padW_ = 0;
    std::vector<int> strides_{8,16,32};

    float nms_thresh_ = 0.45f;
    float bbox_conf_thresh_ = 0.3f;

    int input_w_ = 0;
    int input_h_ = 0;

    std::vector<float> nchw_; // input blob NCHW float

    // ORT-like preprocess (stesso ONNXRuntimeModel::static_resize + blobFromImage)
    cv::Mat static_resize_like_ort(const cv::Mat& img) const;
    void blobFromImage_like_ort(const cv::Mat& img, float* blob_data) const;

    // Postprocess (equivalente a YuNetONNX::postProcess)
    std::vector<Object> postProcess_from_tensors(const std::vector<ov::Tensor>& outs);

    // NMS helpers (copiati da ONNXRuntimeModel per eliminare dipendenza ORT)
    static float intersection_area(const Object &a, const Object &b);
    static void qsort_descent_inplace(std::vector<Object> &objs, int left, int right);
    static void qsort_descent_inplace(std::vector<Object> &objs);
    static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked,
                                  float nms_threshold);
};

} // namespace yunet

