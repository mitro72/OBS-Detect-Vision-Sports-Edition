#include "YuNetOpenVINO.h"
#include <stdexcept>

namespace yunet {

static ov::AnyMap make_config(const std::string& device, int numThreads)
{
    ov::AnyMap cfg;
    cfg.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    if (device == "CPU") {
        cfg.emplace(ov::inference_num_threads(std::max(1, numThreads)));
    }
    return cfg;
}

YuNetOpenVINO::YuNetOpenVINO(const std::string& modelPath,
                             const std::string& device,
                             int numThreads,
                             int keep_topk,
                             float nms_th,
                             float conf_th)
: keep_topk_(keep_topk),
  nms_thresh_(nms_th),
  bbox_conf_thresh_(conf_th)
{
    auto model = core_.read_model(modelPath);

    // ORT assume NCHW, e tu in ORT leggi shape[2],[3]
    auto in = model->input();
    auto shape = in.get_shape(); // [1,3,H,W]
    if (shape.size() != 4)
        throw std::runtime_error("YuNetOpenVINO: input rank != 4");
    input_h_ = (int)shape[2];
    input_w_ = (int)shape[3];

    // padW/padH come nel costruttore YuNetONNX
    padW_ = (int((input_w_ - 1) / divisor_) + 1) * divisor_;
    padH_ = (int((input_h_ - 1) / divisor_) + 1) * divisor_;

    compiled_ = core_.compile_model(model, device, make_config(device, numThreads));
    req_ = compiled_.create_infer_request();

    nchw_.resize(1 * 3 * input_h_ * input_w_);

    // (Opzionale) sanity: YuNet produce tanti output, ma li useremo via get_output_tensor(i)
}

cv::Mat YuNetOpenVINO::static_resize_like_ort(const cv::Mat& img) const
{
    // identico a ONNXRuntimeModel::static_resize con i nostri input_w_/input_h_
    const float r = std::fminf((float)input_w_ / (float)img.cols,
                              (float)input_h_ / (float)img.rows);

    const int unpad_w = (int)(r * (float)img.cols);
    const int unpad_h = (int)(r * (float)img.rows);

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());

    cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114,114,114));
    re.copyTo(out(cv::Rect(0,0,re.cols,re.rows)));
    return out;
}

void YuNetOpenVINO::blobFromImage_like_ort(const cv::Mat& img, float* blob_data) const
{
    // identico a ONNXRuntimeModel::blobFromImage
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

// ==== NMS helpers (copiati da ONNXRuntimeModel) ====
float YuNetOpenVINO::intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void YuNetOpenVINO::qsort_descent_inplace(std::vector<Object> &objs, int left, int right)
{
    int i = left;
    int j = right;
    float p = objs[(left + right) / 2].prob;

    while (i <= j) {
        while (objs[i].prob > p) ++i;
        while (objs[j].prob < p) --j;
        if (i <= j) {
            std::swap(objs[i], objs[j]);
            ++i; --j;
        }
    }
    if (left < j)  qsort_descent_inplace(objs, left, j);
    if (i < right) qsort_descent_inplace(objs, i, right);
}

void YuNetOpenVINO::qsort_descent_inplace(std::vector<Object> &objs)
{
    if (objs.empty()) return;
    qsort_descent_inplace(objs, 0, (int)objs.size() - 1);
}

void YuNetOpenVINO::nms_sorted_bboxes(const std::vector<Object> &objects,
                                      std::vector<int> &picked,
                                      float nms_threshold)
{
    picked.clear();
    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
        areas[i] = objects[i].rect.area();

    for (size_t i = 0; i < n; ++i) {
        const Object &a = objects[i];
        int keep = 1;
        for (size_t j = 0; j < picked.size(); ++j) {
            const Object &b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep) picked.push_back((int)i);
    }
}
// ===============================================

std::vector<Object> YuNetOpenVINO::postProcess_from_tensors(const std::vector<ov::Tensor>& outs)
{
    // YuNetONNX usa: per ogni stride i:
    // cls = result[i]
    // obj = result[i + strides.size()*1]
    // bbox= result[i + strides.size()*2]
    // kps = result[i + strides.size()*3] (commentato)
    const size_t S = strides_.size();
    if (outs.size() < S * 3)
        throw std::runtime_error("YuNetOpenVINO: not enough output tensors");

    std::vector<Object> faces;

    for (size_t i = 0; i < S; ++i) {
        const float stride = (float)strides_[i];
        const int cols = int((float)padW_ / stride);
        const int rows = int((float)padH_ / stride);

        const ov::Tensor& cls  = outs[i];
        const ov::Tensor& obj  = outs[i + S * 1];
        const ov::Tensor& bbox = outs[i + S * 2];

        const float* cls_v = static_cast<const float*>(cls.data());
        const float* obj_v = static_cast<const float*>(obj.data());
        const float* bbox_v = static_cast<const float*>(bbox.data());
for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const size_t idx = (size_t)r * (size_t)cols + (size_t)c;

                const float cls_score = std::clamp(cls_v[idx], 0.f, 1.f);
                const float obj_score = std::clamp(obj_v[idx], 0.f, 1.f);

                Object face;
                face.prob = std::sqrt(cls_score * obj_score);
                if (face.prob < bbox_conf_thresh_)
                    continue;

                const float cx = ((float)c + bbox_v[idx * 4 + 0]) * stride;
                const float cy = ((float)r + bbox_v[idx * 4 + 1]) * stride;
                const float w  = std::exp(bbox_v[idx * 4 + 2]) * stride;
                const float h  = std::exp(bbox_v[idx * 4 + 3]) * stride;

                const float x1 = cx - w / 2.f;
                const float y1 = cy - h / 2.f;

                face.rect = cv::Rect2f(x1, y1, w, h);
                face.label = 0;

                faces.push_back(face);
            }
        }
    }

    // NMS
    qsort_descent_inplace(faces);
    std::vector<int> picked;
    nms_sorted_bboxes(faces, picked, nms_thresh_);

    // Keep topk
    if ((size_t)keep_topk_ < picked.size())
        picked.resize((size_t)keep_topk_);

    std::vector<Object> faces_nms;
    faces_nms.reserve(picked.size());
    for (size_t i = 0; i < picked.size(); ++i)
        faces_nms.push_back(faces[picked[i]]);

    return faces_nms;
}

std::vector<Object> YuNetOpenVINO::inference(const cv::Mat &frame)
{
    if (frame.empty()) return {};

    // preprocess ORT-identico
    cv::Mat pr_img = static_resize_like_ort(frame);
    blobFromImage_like_ort(pr_img, nchw_.data());

    ov::Tensor inTensor(ov::element::f32,
                        ov::Shape{1,3,(size_t)input_h_,(size_t)input_w_},
                        nchw_.data());
    req_.set_input_tensor(inTensor);

    // infer
    req_.infer();

    // raccogli output tensor in ordine indice (come faceva ORT result[i])
    const size_t outCount = compiled_.outputs().size();
    std::vector<ov::Tensor> outs;
    outs.reserve(outCount);
    for (size_t i = 0; i < outCount; ++i)
        outs.push_back(req_.get_output_tensor(i));

    // postprocess
    std::vector<Object> objects = postProcess_from_tensors(outs);

    // scale back (identico al tuo codice)
    const float scale = std::fminf((float)input_w_ / (float)frame.cols,
                                   (float)input_h_ / (float)frame.rows);

    for (auto &obj : objects) {
        obj.rect.x      /= scale;
        obj.rect.y      /= scale;
        obj.rect.width  /= scale;
        obj.rect.height /= scale;
    }
    return objects;
}

} // namespace yunet
