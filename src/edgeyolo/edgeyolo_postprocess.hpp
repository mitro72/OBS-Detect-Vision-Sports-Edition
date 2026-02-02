#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>

// Ti serve Object + nms/sort helpers
#include "ort-model/types.hpp"   // Object
#include <cmath>

namespace edgeyolo_cpp {

inline void generate_edgeyolo_proposals(int num_array,
                                        const float *feat_ptr,
                                        float prob_threshold,
                                        int num_classes,
                                        std::vector<Object> &objects)
{
    for (int idx = 0; idx < num_array; ++idx) {
        const int basic_pos = idx * (num_classes + 5);

        float box_objectness = feat_ptr[basic_pos + 4];
        int class_id = 0;
        float max_class_score = 0.0f;

        for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
            float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > max_class_score) {
                class_id = class_idx;
                max_class_score = box_prob;
            }
        }

        if (max_class_score > prob_threshold) {
            float x_center = feat_ptr[basic_pos + 0];
            float y_center = feat_ptr[basic_pos + 1];
            float w        = feat_ptr[basic_pos + 2];
            float h        = feat_ptr[basic_pos + 3];

            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width  = w;
            obj.rect.height = h;
            obj.label = class_id;
            obj.prob  = max_class_score;
            objects.push_back(obj);
        }
    }
}

inline float intersection_area(const Object &a, const Object &b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

inline void qsort_descent_inplace(std::vector<Object> &objs, int left, int right)
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

inline void qsort_descent_inplace(std::vector<Object> &objs)
{
	if (objs.empty()) return;
	qsort_descent_inplace(objs, 0, (int)objs.size() - 1);
}

inline void nms_sorted_bboxes(const std::vector<Object> &objects,
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

inline void decode_outputs(const float *prob,
                           int num_array,
                           std::vector<Object> &objects,
                           float bbox_conf_thresh,
                           float nms_thresh,
                           int num_classes,
                           float scale,
                           int img_w,
                           int img_h)
{
    std::vector<Object> proposals;
    generate_edgeyolo_proposals(num_array, prob, bbox_conf_thresh, num_classes, proposals);

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    objects.clear();
    for (int i = 0; i < (int)picked.size(); ++i) {
        Object obj = proposals[picked[i]];

        float x0 = obj.rect.x / scale;
        float y0 = obj.rect.y / scale;
        float x1 = (obj.rect.x + obj.rect.width) / scale;
        float y1 = (obj.rect.y + obj.rect.height) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.id = (uint64_t)objects.size() + 1;

        objects.push_back(obj);
    }
}

} // namespace edgeyolo_cpp
