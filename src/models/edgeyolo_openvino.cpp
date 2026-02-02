#include "models/edgeyolo_openvino.hpp"
#include "detect-filter-utils.h"   // se contiene helper utili
#include "ort-model/utils.hpp"     // se contiene decode/NMS riusabili (nome fuorviante ma ok)
#include <stdexcept>

namespace edgeyolo_cpp {

static ov::AnyMap make_config(const std::string& device, int numThreads) {
    ov::AnyMap cfg;
    if (device == "CPU") {
        // Threading CPU
        cfg.emplace(ov::inference_num_threads(numThreads));
        cfg.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    } else {
        // GPU (Intel): spesso non usa inference_num_threads
        cfg.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    }
    // Cache compilation (facoltativo, utile per startup)
    // cfg.emplace(ov::cache_dir("..."));
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

    // Input shape: leggiamo dal modello
    input_ = model->input();
    auto shape = input_.get_shape(); // es: [1,3,H,W] o [1,H,W,3]
    if (shape.size() != 4)
        throw std::runtime_error("Unexpected input rank (expected 4D).");

    // Determina H/W in base al layout
    // YOLO spesso NCHW: [N,C,H,W]
    // Alcuni modelli possono essere NHWC.
    if (shape[1] == 3) { // NCHW
        inH_ = (int)shape[2];
        inW_ = (int)shape[3];
    } else if (shape[3] == 3) { // NHWC
        inH_ = (int)shape[1];
        inW_ = (int)shape[2];
    } else {
        throw std::runtime_error("Cannot infer model input layout.");
    }

    // Compila
    compiled_ = core_.compile_model(model, device, make_config(device, numThreads));
    req_ = compiled_.create_infer_request();

    output_ = compiled_.output();
}

std::vector<Object> EdgeYOLOOpenVINO::inference(const cv::Mat& bgr)
{
    if (bgr.empty())
        return {};

    std::lock_guard<std::mutex> lock(mtx_);

    // 1) Preprocess: resize/letterbox coerente col tuo decoder
    // Qui ti metto un resize semplice; MA per YOLO serve tipicamente letterbox.
    // >>> Qui riusiamo il tuo letterbox esistente (me lo prendo dai file edgeyolo).
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inW_, inH_));

    // 2) Converti a FP32 NCHW
    cv::Mat f32;
    resized.convertTo(f32, CV_32F, 1.0 / 255.0);

    std::vector<float> nchw(1 * 3 * inH_ * inW_);
    // HWC -> CHW
    for (int y = 0; y < inH_; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < inW_; ++x) {
            const cv::Vec3f v = row[x]; // BGR
            // CHW
            nchw[0 * inH_ * inW_ + y * inW_ + x] = v[0];
            nchw[1 * inH_ * inW_ + y * inW_ + x] = v[1];
            nchw[2 * inH_ * inW_ + y * inW_ + x] = v[2];
        }
    }

    ov::Tensor inputTensor(ov::element::f32, ov::Shape{1, 3, (size_t)inH_, (size_t)inW_}, nchw.data());
    req_.set_input_tensor(inputTensor);

    // 3) Infer
    req_.infer();

    // 4) Output tensor
    ov::Tensor out = req_.get_output_tensor();
    const float* outData = out.data<const float>();
    const ov::Shape outShape = out.get_shape();

    // 5) Decode + NMS
    // >>> QUI va collegato il tuo decoder EdgeYOLO (che oggi vive in EdgeYOLOONNXRuntime).
    // L’obiettivo è produrre: std::vector<Object>
    //
    // Esempio (placeholder):
    std::vector<Object> objects;
    // decode_edgeyolo(outData, outShape, confTh_, nmsTh_, numClasses_, objects, ...);

    return objects;
}

} // namespace edgeyolo_cpp
