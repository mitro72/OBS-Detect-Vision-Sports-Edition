#include "edgeyolo_openvino.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace edgeyolo_cpp {

static ov::AnyMap make_config(const std::string& device, int numThreads)
{
    ov::AnyMap cfg;
    cfg.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    if (device == "CPU") {
        cfg.emplace(ov::inference_num_threads(std::max(1, numThreads)));
    }
    return cfg;
}

EdgeYOLOOpenVINO::EdgeYOLOOpenVINO(const std::string& modelPath,
                                   const std::string& device,
                                   int numThreads,
                                   int numClasses,
                                   float nmsTh,
                                   float confTh)
: numClasses_(numClasses), nmsTh_(nmsTh), confTh_(confTh)
{
    auto model = core_.read_model(modelPath);

    // ORT assume NCHW
    auto in = model->input();
    auto shape = in.get_shape(); // [1,3,H,W]
    if (shape.size() != 4)
        throw std::runtime_error("EdgeYOLOOpenVINO: input rank != 4");
    inH_ = (int)shape[2];
    inW_ = (int)shape[3];

    compiled_ = core_.compile_model(model, device, make_config(device, numThreads));
    req_ = compiled_.create_infer_request();

    // output size -> numArray = total / (5+numClasses)
    auto out = compiled_.output();
    auto osh = out.get_shape();
    size_t total = 1;
    for (auto d : osh) total *= (size_t)d;

    const int stride = 5 + numClasses_;
    if (stride <= 0 || (total % (size_t)stride) != 0)
        throw std::runtime_error("EdgeYOLOOpenVINO: output not divisible by (5+numClasses)");

    numArray_ = (int)(total / (size_t)stride);

    // pre-alloc input buffer
    nchw_.resize(1 * 3 * inH_ * inW_);
}

cv::Mat EdgeYOLOOpenVINO::static_resize_like_ort(const cv::Mat& img) const
{
    // Copia 1:1 da ONNXRuntimeModel::static_resize (pad right/bottom, origin (0,0))
    const float r = std::fminf((float)inW_ / (float)img.cols,
                              (float)inH_ / (float)img.rows);

    const int unpad_w = (int)(r * (float)img.cols);
    const int unpad_h = (int)(r * (float)img.rows);

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());

    cv::Mat out(inH_, inW_, CV_8UC3, cv::Scalar(114,114,114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void EdgeYOLOOpenVINO::blobFromImage_like_ort(const cv::Mat& img, float* blob_data) const
{
    // Copia 1:1 da ONNXRuntimeModel::blobFromImage (BGR uint8 -> float NCHW, NO /255)
    const int H = img.rows;
    const int W = img.cols;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < H; ++h) {
            const cv::Vec3b* row = img.ptr<cv::Vec3b>(h);
            for (int w = 0; w < W; ++w) {
                blob_data[c * W * H + h * W + w] = (float)row[w][c];
            }
        }
    }
}

std::vector<Object> EdgeYOLOOpenVINO::inference(const cv::Mat& frame)
{
    if (frame.empty()) return {};

    // preprocess ORT-identico
    cv::Mat pr_img = static_resize_like_ort(frame);
    blobFromImage_like_ort(pr_img, nchw_.data());

    // infer
    ov::Tensor inTensor(ov::element::f32,
                        ov::Shape{1,3,(size_t)inH_,(size_t)inW_},
                        nchw_.data());
    req_.set_input_tensor(inTensor);
    req_.infer();

    ov::Tensor out = req_.get_output_tensor();
    const float* net_pred = out.data<const float>();

    // postprocess ORT-identico
    float scale = std::fminf((float)inW_ / (float)frame.cols,
                             (float)inH_ / (float)frame.rows);

    std::vector<Object> objects;
    decode_outputs(net_pred, numArray_, objects, confTh_, nmsTh_, numClasses_,
                   scale, frame.cols, frame.rows);
    return objects;
}

} // namespace edgeyolo_cpp

