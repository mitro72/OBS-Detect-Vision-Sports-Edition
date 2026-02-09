#include "detect-filter.h"

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#include <util/platform.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "models/OpenVINOAdapters.h"
#include "edgeyolo/coco_names.hpp"
#include "yunet/YuNetOpenVINO.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define FACE_DETECT_MODEL_SIZE "!!!FACE_DETECT!!!"

#ifdef _WIN32
static std::string wide_to_utf8(const std::wstring& w)
{
	if (w.empty()) return {};
	int size = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(size, 0);
	WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), size, nullptr, nullptr);
	return s;
}
#endif



static inline float smooth_damp_critically_damped(float current, float target, float &currentVelocity,
						  float smoothTime, float deltaTime)
{
	// Based on a critically damped spring (Unity-like SmoothDamp). Stable for variable deltaTime.
	smoothTime = std::max(0.0001f, smoothTime);
	const float omega = 2.0f / smoothTime;

	const float x = omega * deltaTime;
	// Stable approximation of exp(-omega * dt)
	const float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);

	const float change = current - target;
	const float temp = (currentVelocity + omega * change) * deltaTime;
	currentVelocity = (currentVelocity - omega * temp) * exp;

	const float output = target + (change + temp) * exp;
	return output;
}

static inline float smooth_time_from_alpha60(float alpha_per_frame_60fps)
{
	// Convert a "lerp alpha per frame at ~60fps" into an equivalent smoothTime (seconds)
	// for a critically damped spring. This keeps the user-facing "zoomSpeedFactor" feeling similar.
	const float a = std::clamp(alpha_per_frame_60fps, 0.0f, 0.999999f);
	if (a <= 0.0f)
		return 1000000.0f; // effectively frozen
	const float dt_ref = 1.0f / 60.0f;
	const float k = -logf(1.0f - a) / dt_ref; // first-order equivalent (1/s)
	return std::max(0.0001f, 2.0f / k);        // map to spring smoothTime via omega=2/smoothTime
}


struct RectI { int x, y, w, h; };

static inline RectI make_safe_roi(int w, int h, int lPct, int rPct, int tPct, int bPct)
{
	const int l = (w * lPct) / 100;
	const int r = (w * rPct) / 100;
	const int t = (h * tPct) / 100;
	const int b = (h * bPct) / 100;

	RectI rc;
	rc.x = l;
	rc.y = t;
	rc.w = std::max(1, w - l - r);
	rc.h = std::max(1, h - t - b);
	return rc;
}

static inline bool point_in_rect(float px, float py, const RectI &r)
{
	return px >= (float)r.x && py >= (float)r.y && px < (float)(r.x + r.w) && py < (float)(r.y + r.h);
}

static inline bool obj_center_in_safe(const Object &obj, const RectI &safe)
{
	const float cx = obj.rect.x + obj.rect.width * 0.5f;
	const float cy = obj.rect.y + obj.rect.height * 0.5f;
	return point_in_rect(cx, cy, safe);
}

static inline bool rect_valid(const cv::Rect2f &r)
{
	return r.width > 1.0f && r.height > 1.0f;
}


static inline float rect_center_dist(const cv::Rect2f &a, const cv::Rect2f &b)
{
	const float ax = a.x + a.width * 0.5f;
	const float ay = a.y + a.height * 0.5f;
	const float bx = b.x + b.width * 0.5f;
	const float by = b.y + b.height * 0.5f;
	const float dx = ax - bx;
	const float dy = ay - by;
	return std::sqrt(dx * dx + dy * dy);
}



struct detect_filter : public filter_data {
	// SmoothDamp velocities for trackingRect (x, y, w, h)
	float trackVelX = 0.0f;
	float trackVelY = 0.0f;
	float trackVelW = 0.0f;
	float trackVelH = 0.0f;
	int groupMinPeople = 6;
	bool groupMinPeopleStrict = false;
	float groupMaxDistFrac = 0.15f; // max distance between people (fraction of frame width)
	bool previewGroupClusters = false;
	bool previewGroupClusterLabel = false;

	// Safe ROI (decision region) margins in percent (used only for crop decision, not inference)
	int safe_roi_left = 10;
	int safe_roi_right = 10;
	int safe_roi_top = 0;
	int safe_roi_bottom = 8;

	// Safe ROI hold-before-fallback (ms)
	int safe_roi_hold_ms = 300;
	int safe_roi_hold_timer_ms = 0;
	bool safe_roi_holding = false;
	uint64_t safe_roi_hold_until_ns = 0;
	cv::Rect2f safe_roi_last_good_bbox = cv::Rect2f(0, 0, 0, 0);

	// Debug/preview: bbox used to drive crop decision
	cv::Rect2f safe_roi_decision_bbox = cv::Rect2f(0, 0, 0, 0);
	bool safe_roi_decision_from_safe = false;
	bool safe_roi_decision_is_hold = false;


	// Cluster temporal inertia (ms): delay before switching to a different best cluster
	int cluster_inertia_ms = 150;

	uint64_t cluster_pending_since_ns = 0;
	cv::Rect2f cluster_pending_box = cv::Rect2f(0, 0, 0, 0);

	cv::Rect2f cluster_active_box = cv::Rect2f(0, 0, 0, 0);
	bool cluster_active_valid = false;

	// Debug/preview: cluster inertia state
	bool cluster_inertia_pending = false;


	// Cache/throttle for group bbox selection (crop)
	uint64_t lastGroupBoxTsNs = 0;
	cv::Rect2f lastGroupBox;
	int lastGroupCount = 0;
	bool lastGroupBoxValid = false;

	// CPU OPT: inference throttling + caching
	int infer_interval_ms = 0;          // 0 = disabled (infer every frame)
	float infer_scale = 1.0f;           // 1.0 = disabled
	uint64_t last_infer_ts_ns = 0;
	bool cached_objects_valid = false;
	std::vector<Object> cached_objects;

	// Horizontal pan preset for group mode: "auto", "left", "center", "right"
	std::string x_pan_preset = "auto";
	// Auto-snap state (0=left,1=center,2=right) for "autosnap"
	int x_snap_state = 1;
	// Transition time (seconds) for smooth auto-snap movement ("autosnap_smooth")
	float x_snap_transition_time = 0.25f;
	// Hysteresis margin for autosnap in normalized units (0..0.2 typical)
	float x_snap_hysteresis = 0.05f;
	// Deadband on target X (percent of frame width). 0 disables.
	float x_deadband = 0.0f;
	float last_target_zx = 0.0f;
	bool has_last_target_zx = false;
};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p,
				     obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");

	for (const char *prop_name :
	     {"threshold", "useGPU", "numThreads", "model_size", "detected_object", "sort_tracking",
	      "max_unseen_frames", "show_unseen_objects", "save_detections_path", "crop_group",
	      "min_size_threshold"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	return true;
}

void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	// sort the vector based on the class names
	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	// read the '.json' file near the model file to find the class names
	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		// parse the JSON file
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	// add dropdown selection for object category selection: "All", or COCO classes
	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);
	tf->classNames = edgeyolo_cpp::COCO_CLASSES;

	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								  obs_property_t *,
								  obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_t *prop = obs_properties_get(props_, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");

		obs_property_set_visible(prop, enabled);
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		obs_property_set_visible(masking_dilation, enabled);
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, enabled);
		}
		return true;
	});

	// add masking options drop down selection: "None", "Solid color", "Blur", "Transparent"
	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
							       obs_module_text("MaskingType"),
							       OBS_COMBO_TYPE_LIST,
							       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(masking_type, obs_module_text("None"), "none");
	obs_property_list_add_string(masking_type, obs_module_text("SolidColor"), "solid_color");
	obs_property_list_add_string(masking_type, obs_module_text("OutputMask"), "output_mask");
	obs_property_list_add_string(masking_type, obs_module_text("Blur"), "blur");
	obs_property_list_add_string(masking_type, obs_module_text("Pixelate"), "pixelate");
	obs_property_list_add_string(masking_type, obs_module_text("Transparent"), "transparent");

	// add color picker for solid color masking
	obs_properties_add_color(masking_group, "masking_color", obs_module_text("MaskingColor"));

	// add slider for blur radius
	obs_properties_add_int_slider(masking_group, "masking_blur_radius",
				      obs_module_text("MaskingBlurRadius"), 1, 30, 1);

	// add callback to show/hide blur radius and color picker
	obs_property_set_modified_callback(masking_type, [](obs_properties_t *props_,
							    obs_property_t *,
							    obs_data_t *settings) {
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_set_visible(masking_dilation, masking_enabled);

		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, masking_enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, masking_enabled);
		}
		return true;
	});

	// add slider for dilation iterations
	obs_properties_add_int_slider(masking_group, "dilation_iterations",
				      obs_module_text("DilationIterations"), 0, 20, 1);

	// add options group for tracking and zoom-follow options
	obs_properties_t *tracking_group_props = obs_properties_create();
	obs_property_t *tracking_group = obs_properties_add_group(
		props, "tracking_group", obs_module_text("TrackingZoomFollowGroup"),
		OBS_GROUP_CHECKABLE, tracking_group_props);

	// add callback to show/hide tracking options
	obs_property_set_modified_callback(tracking_group, [](obs_properties_t *props_,
							      obs_property_t *,
							      obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "tracking_group");

		// Show/hide the core tracking controls when the group is enabled/disabled
		for (auto prop_name : {"zoom_factor", "zoom_object", "zoom_speed_factor", "x_pan_preset",
				       "x_snap_hysteresis", "x_snap_transition_time", "x_deadband", "infer_interval_ms", "infer_scale",
				       "group_min_people", "group_min_people_strict", "safe_roi_left", "safe_roi_right", "safe_roi_top", "safe_roi_bottom", "safe_roi_hold_ms", "cluster_inertia_ms"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			if (prop)
				obs_property_set_visible(prop, enabled);
		}

		// Cluster preview controls only make sense when ZoomObject == "group"
		const char *zo = obs_data_get_string(settings, "zoom_object");
		const bool is_group = (zo && strcmp(zo, "group") == 0);

		obs_property_t *gmp = obs_properties_get(props_, "group_min_people");
		if (gmp)
			obs_property_set_visible(gmp, enabled && is_group);

		obs_property_t *gms = obs_properties_get(props_, "group_min_people_strict");
		if (gms)
			obs_property_set_visible(gms, enabled && is_group);

		obs_property_t *gmd = obs_properties_get(props_, "group_max_dist_frac");
		if (gmd)
			obs_property_set_visible(gmd, enabled && is_group);


		obs_property_t *pgc = obs_properties_get(props_, "preview_group_clusters");
		if (pgc)
			obs_property_set_visible(pgc, enabled && is_group);

		obs_property_t *lbl = obs_properties_get(props_, "preview_group_cluster_label");
		if (lbl) {
			const bool on = obs_data_get_bool(settings, "preview_group_clusters");
			obs_property_set_visible(lbl, enabled && is_group && on);
		}

		return true;
	});

	// add zoom factor slider
	obs_properties_add_float_slider(tracking_group_props, "zoom_factor",
					obs_module_text("ZoomFactor"), 0.0, 1.0, 0.05);

	obs_properties_add_float_slider(tracking_group_props, "zoom_speed_factor",
					obs_module_text("ZoomSpeed"), 0.0, 0.1, 0.01);

	// Group pan preset: manual end-stops left/center/right (or auto follow)
	obs_property_t *x_pan_preset = obs_properties_add_list(tracking_group_props, "x_pan_preset",
							 obs_module_text("XPosition"),
							 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(x_pan_preset, obs_module_text("Auto"), "auto");
	obs_property_list_add_string(x_pan_preset, obs_module_text("Left"), "left");
	obs_property_list_add_string(x_pan_preset, obs_module_text("Center"), "center");
	obs_property_list_add_string(x_pan_preset, obs_module_text("Right"), "right");
	obs_property_list_add_string(x_pan_preset, obs_module_text("Auto Snap"), "autosnap");
	obs_property_list_add_string(x_pan_preset, obs_module_text("Auto Snap (Smooth)"), "autosnap_smooth");

	obs_properties_add_float_slider(tracking_group_props, "x_snap_hysteresis",
					 obs_module_text("Snap Hysteresis"), 0.0, 0.20, 0.01);

	obs_properties_add_float_slider(tracking_group_props, "x_snap_transition_time",
					 obs_module_text("Snap Transition (s)"), 0.05, 1.00, 0.05);
	
	obs_properties_add_int_slider(tracking_group_props, "infer_interval_ms",
			      obs_module_text("Infer Interval (ms)"), 0, 200, 5);

	obs_properties_add_float_slider(tracking_group_props, "infer_scale",
				obs_module_text("Infer Scale"), 0.25, 1.00, 0.05);

	
	obs_properties_add_float_slider(tracking_group_props, "x_deadband",
				obs_module_text("X Deadband (%)"), 0.0, 5.0, 0.1);

	// add object selection for zoom drop down: "Single", "All"
	obs_property_t *zoom_object = obs_properties_add_list(tracking_group_props, "zoom_object",
							      obs_module_text("ZoomObject"),
							      OBS_COMBO_TYPE_LIST,
							      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(zoom_object, obs_module_text("SingleFirst"), "single");
	obs_property_list_add_string(zoom_object, obs_module_text("Biggest"), "biggest");
	obs_property_list_add_string(zoom_object, obs_module_text("Oldest"), "oldest");
	obs_property_list_add_string(zoom_object, obs_module_text("All"), "all");
	//mod x basket
	obs_property_list_add_string(zoom_object, obs_module_text("Group"), "group");
obs_property_set_modified_callback(zoom_object, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		// When switching ZoomObject, show cluster preview controls only for "group"
		const bool enabled = obs_data_get_bool(settings, "tracking_group");
		const char *zo = obs_data_get_string(settings, "zoom_object");
		const bool is_group = (zo && strcmp(zo, "group") == 0);

		obs_property_t *pgc = obs_properties_get(props_, "preview_group_clusters");
		if (pgc)
			obs_property_set_visible(pgc, enabled && is_group);

		obs_property_t *lbl = obs_properties_get(props_, "preview_group_cluster_label");
		if (lbl) {
			const bool on = obs_data_get_bool(settings, "preview_group_clusters");
			obs_property_set_visible(lbl, enabled && is_group && on);
		}
		return true;
	});

	obs_properties_add_int(tracking_group_props, "group_min_people", "Group min people", 1, 15, 1);
obs_properties_add_float_slider(tracking_group_props, "group_max_dist_frac", obs_module_text("GroupMaxDistFrac"), 0.05, 0.50, 0.01);

	obs_properties_add_int(tracking_group_props, "safe_roi_left", "Safe ROI Left Margin (%)", 0, 40, 1);
	obs_properties_add_int(tracking_group_props, "safe_roi_right", "Safe ROI Right Margin (%)", 0, 40, 1);
	obs_properties_add_int(tracking_group_props, "safe_roi_top", "Safe ROI Top Margin (%)", 0, 40, 1);
	obs_properties_add_int(tracking_group_props, "safe_roi_bottom", "Safe ROI Bottom Margin (%)", 0, 40, 1);
	obs_properties_add_int(tracking_group_props, "safe_roi_hold_ms", "Safe ROI Hold (ms)", 0, 2000, 50);
	obs_properties_add_int_slider(tracking_group_props, "cluster_inertia_ms", "Cluster inertia (ms)", 0, 2000, 25);
		obs_properties_add_bool(tracking_group_props, "group_min_people_strict", "Strict min people");
	obs_property_t *preview_group_clusters =
		obs_properties_add_bool(tracking_group_props, "preview_group_clusters", "Preview group cluster");
	obs_property_t *preview_group_cluster_label =
		obs_properties_add_bool(tracking_group_props, "preview_group_cluster_label", "Show group cluster label");
	// Initial visibility: detect_filter_properties() has no access to current settings -> start hidden.
	obs_property_set_visible(preview_group_clusters, false);
	obs_property_set_visible(preview_group_cluster_label, false);
	// Show label option only when cluster preview is enabled
	obs_property_set_modified_callback(preview_group_clusters, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "tracking_group");
		const bool on = obs_data_get_bool(settings, "preview_group_clusters");
		const char *zo = obs_data_get_string(settings, "zoom_object");
		const bool is_group = (zo && strcmp(zo, "group") == 0);
		obs_property_t *lbl = obs_properties_get(props_, "preview_group_cluster_label");
		if (lbl)
			obs_property_set_visible(lbl, enabled && is_group && on);
		return true;
	});

	obs_property_t *advanced =
		obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));

	// If advanced is selected show the advanced settings, otherwise hide them
	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	// add a checkable group for crop region settings
	obs_properties_t *crop_group_props = obs_properties_create();
	obs_property_t *crop_group =
		obs_properties_add_group(props, "crop_group", obs_module_text("CropGroup"),
					 OBS_GROUP_CHECKABLE, crop_group_props);

	// add callback to show/hide crop region options
	obs_property_set_modified_callback(crop_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "crop_group");
		for (auto prop_name : {"crop_left", "crop_right", "crop_top", "crop_bottom"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add crop region settings
	obs_properties_add_int_slider(crop_group_props, "crop_left", obs_module_text("CropLeft"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_right", obs_module_text("CropRight"),
				      0, 1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_top", obs_module_text("CropTop"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_bottom",
				      obs_module_text("CropBottom"), 0, 1000, 1);

	// add a text input for the currently detected object
	obs_property_t *detected_obj_prop = obs_properties_add_text(
		props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	// disable the text input by default
	obs_property_set_enabled(detected_obj_prop, false);

	// add threshold slider
	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	// add minimal size threshold slider
	obs_properties_add_int_slider(props, "min_size_threshold",
				      obs_module_text("MinSizeThreshold"), 0, 10000, 1);

	// add SORT tracking enabled checkbox
	obs_properties_add_bool(props, "sort_tracking", obs_module_text("SORTTracking"));

	// add parameter for number of missing frames before a track is considered lost
	obs_properties_add_int(props, "max_unseen_frames", obs_module_text("MaxUnseenFrames"), 1,
			       30, 1);

	// add option to show unseen objects
	obs_properties_add_bool(props, "show_unseen_objects", obs_module_text("ShowUnseenObjects"));

	// add file path for saving detections
	obs_properties_add_path(props, "save_detections_path",
				obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE_SAVE,
				"JSON file (*.json);;All files (*.*)", nullptr);

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
	// OpenVINO
	obs_property_list_add_string(p_use_gpu, obs_module_text("OpenVINOCPU"), USEGPU_OV_CPU);
	obs_property_list_add_string(p_use_gpu, obs_module_text("OpenVINOGPU"), USEGPU_OV_GPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"), USEGPU_TENSORRT);
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUCuda"), USEGPU_CUDA);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"), USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 16, 1);

	// add drop down option for model size: Small, Medium, Large
	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model_size, obs_module_text("SmallFast"), "small");
	obs_property_list_add_string(model_size, obs_module_text("Medium"), "medium");
	obs_property_list_add_string(model_size, obs_module_text("LargeSlow"), "large");
	obs_property_list_add_string(model_size, obs_module_text("FaceDetect"),
				     FACE_DETECT_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	// add external model file path
	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "EdgeYOLO onnx files (*.onnx);;all files (*.*)",
				nullptr);

	// add callback to show/hide the external model file path
	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);
			if (!is_external) {
				if (model_size_value == FACE_DETECT_MODEL_SIZE) {
					// set the class names to COCO classes for face detection model
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						yunet::FACE_CLASSES);
					tf_->classNames = yunet::FACE_CLASSES;
				} else {
					// reset the class names to COCO classes for default models
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						edgeyolo_cpp::COCO_CLASSES);
					tf_->classNames = edgeyolo_cpp::COCO_CLASSES;
				}
			} else {
				// if the model path is already set - update the class names
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings, tf_);
			}
			return true;
		},
		tf);

	// add callback on the model file path to check if the file exists
	obs_property_set_modified_callback2(
		obs_properties_get(props, "external_model_file"),
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(model_size_value, EXTERNAL_MODEL_SIZE) == 0;
			if (!is_external) {
				return true;
			}
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings,
								   tf_);
			return true;
		},
		tf);

	// Add a informative text about the plugin
	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", false);
	obs_data_set_default_int(settings, "max_unseen_frames", 10);
	obs_data_set_default_bool(settings, "show_unseen_objects", true);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", "small");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_bool(settings, "masking_group", false);
	obs_data_set_default_string(settings, "masking_type", "none");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 0);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_double(settings, "zoom_factor", 0.0);
	obs_data_set_default_double(settings, "zoom_speed_factor", 0.05);
	obs_data_set_default_string(settings, "zoom_object", "single");
	obs_data_set_default_string(settings, "x_pan_preset", "auto");
	obs_data_set_default_int(settings, "infer_interval_ms", 0);
	obs_data_set_default_double(settings, "infer_scale", 1.0);
	obs_data_set_default_int(settings, "group_min_people", 6);
	obs_data_set_default_bool(settings, "group_min_people_strict", false);
	obs_data_set_default_double(settings, "group_max_dist_frac", 0.15);
	obs_data_set_default_int(settings, "safe_roi_left", 10);
	obs_data_set_default_int(settings, "safe_roi_right", 10);
	obs_data_set_default_int(settings, "safe_roi_top", 0);
	obs_data_set_default_int(settings, "safe_roi_bottom", 8);
	obs_data_set_default_int(settings, "safe_roi_hold_ms", 300);
	obs_data_set_default_int(settings, "cluster_inertia_ms", 150);
	obs_data_set_default_bool(settings, "preview_group_clusters", false);
	obs_data_set_default_bool(settings, "preview_group_cluster_label", false);
	obs_data_set_default_double(settings, "x_snap_hysteresis", 0.05);
	obs_data_set_default_double(settings, "x_snap_transition_time", 0.25);
	obs_data_set_default_double(settings, "x_deadband", 0.0);
	obs_data_set_default_string(settings, "save_detections_path", "");
	obs_data_set_default_bool(settings, "crop_group", false);
	obs_data_set_default_int(settings, "crop_left", 0);
	obs_data_set_default_int(settings, "crop_right", 0);
	obs_data_set_default_int(settings, "crop_top", 0);
	obs_data_set_default_int(settings, "crop_bottom", 0);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "dilation_iterations");

	bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");

	tf->groupMinPeople = (int)obs_data_get_int(settings, "group_min_people");
	tf->groupMinPeople = std::max(1, tf->groupMinPeople);
	tf->groupMinPeopleStrict = obs_data_get_bool(settings, "group_min_people_strict");
	tf->groupMaxDistFrac = (float)obs_data_get_double(settings, "group_max_dist_frac");
	tf->groupMaxDistFrac = std::clamp(tf->groupMaxDistFrac, 0.05f, 0.50f);

	tf->safe_roi_left = (int)obs_data_get_int(settings, "safe_roi_left");
	tf->safe_roi_right = (int)obs_data_get_int(settings, "safe_roi_right");
	tf->safe_roi_top = (int)obs_data_get_int(settings, "safe_roi_top");
	tf->safe_roi_bottom = (int)obs_data_get_int(settings, "safe_roi_bottom");
	tf->safe_roi_hold_ms = (int)obs_data_get_int(settings, "safe_roi_hold_ms");
	tf->cluster_inertia_ms = (int)obs_data_get_int(settings, "cluster_inertia_ms");

	tf->safe_roi_left = std::clamp(tf->safe_roi_left, 0, 40);
	tf->safe_roi_right = std::clamp(tf->safe_roi_right, 0, 40);
	tf->safe_roi_top = std::clamp(tf->safe_roi_top, 0, 40);
	tf->safe_roi_bottom = std::clamp(tf->safe_roi_bottom, 0, 40);
	tf->safe_roi_hold_ms = std::clamp(tf->safe_roi_hold_ms, 0, 2000);
	tf->cluster_inertia_ms = std::clamp(tf->cluster_inertia_ms, 0, 2000);
	tf->previewGroupClusters = obs_data_get_bool(settings, "preview_group_clusters");
	tf->previewGroupClusterLabel = obs_data_get_bool(settings, "preview_group_cluster_label");

	tf->x_pan_preset = obs_data_get_string(settings, "x_pan_preset");
	tf->x_snap_hysteresis = (float)obs_data_get_double(settings, "x_snap_hysteresis");
	tf->x_snap_transition_time = (float)obs_data_get_double(settings, "x_snap_transition_time");
	tf->infer_interval_ms = (int)obs_data_get_int(settings, "infer_interval_ms");
	tf->infer_interval_ms = std::clamp(tf->infer_interval_ms, 0, 200);

	tf->infer_scale = (float)obs_data_get_double(settings, "infer_scale");
	tf->infer_scale = std::clamp(tf->infer_scale, 0.25f, 1.0f);

	// reset cache on settings change
	tf->cached_objects_valid = false;
	tf->last_infer_ts_ns = 0;

	tf->x_deadband = (float)obs_data_get_double(settings, "x_deadband");
	tf->x_deadband = std::clamp(tf->x_deadband, 0.0f, 5.0f);
	tf->has_last_target_zx = false; // reset safe when settings change


	if (tf->x_pan_preset == "left")
		tf->x_snap_state = 0;
	else if (tf->x_pan_preset == "center")
		tf->x_snap_state = 1;
	else if (tf->x_pan_preset == "right")
		tf->x_snap_state = 2;

	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");

	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");

	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");

	// check if tracking state has changed
	if (tf->trackingEnabled != newTrackingEnabled) {
		tf->trackingEnabled = newTrackingEnabled;
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (!parent) {
			obs_log(LOG_ERROR, "Parent source not found");
			return;
		}
		if (tf->trackingEnabled) {
			obs_log(LOG_DEBUG, "Tracking enabled");
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
		} else {
			obs_log(LOG_DEBUG, "Tracking disabled");
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (crop_pad_filter) {
				obs_source_filter_remove(parent, crop_pad_filter);
			}
			tf->trackingFilter = nullptr;
		}
	}

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads || tf->modelSize != newModelSize);

	if (reinitialize) {
		obs_log(LOG_INFO, "Reinitializing model");

		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == "small") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_256x416.onnx");
		} else if (newModelSize == "medium") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_480x800.onnx");
		} else if (newModelSize == "large") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_736x1280.onnx");
		} else if (newModelSize == FACE_DETECT_MODEL_SIZE) {
			modelFilepath_rawPtr =
				obs_module_file("models/face_detection_yunet_2023mar.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (!external_model_file || external_model_file[0] == '\0') {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (!modelFilepath_rawPtr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
					    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		// External model labels
		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (!labelsFile.is_open()) {
				obs_log(LOG_ERROR, "Failed to open JSON file for external model labels");
				tf->isDisabled = true;
				tf->model.reset();
				return;
			}
			nlohmann::json j;
			labelsFile >> j;
			if (!j.contains("names")) {
				obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
				tf->isDisabled = true;
				tf->model.reset();
				return;
			}
			std::vector<std::string> labels = j["names"];
			num_classes_ = (int)labels.size();
			tf->classNames = labels;
		} else if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
			num_classes_ = 1;
			tf->classNames = yunet::FACE_CLASSES;
		}

		try {
#ifdef _WIN32
			std::string modelPathString = wide_to_utf8(tf->modelFilepath);
#else
			std::string modelPathString = tf->modelFilepath;
#endif
			const std::string device =
				(tf->useGPU == USEGPU_OV_GPU) ? "GPU" : "CPU";

			tf->model.reset();

			if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
				tf->model = std::make_unique<YuNetOpenVINOAdapter>(
					modelPathString, device, (int)tf->numThreads,
					50, nms_th_, tf->conf_threshold);
			} else {
				tf->model = std::make_unique<EdgeYOLOOpenVINOAdapter>(
					modelPathString, device, (int)tf->numThreads,
					num_classes_, nms_th_, tf->conf_threshold);
			}
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load OpenVINO model: %s", e.what());
			tf->isDisabled = true;
			tf->model.reset();
			return;
		}
	}

	if (tf->model) {
		tf->model->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		obs_log(LOG_INFO, "Detect Filter Options:");
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;

	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
		{MASKING_EFFECT_PATH, &tf->maskingEffect},
		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
	};

	for (auto [effectPath, effect] : effects) {
		char *effectPathPtr = obs_module_file(effectPath);
		if (!effectPathPtr) {
			obs_log(LOG_ERROR, "Failed to get effect path: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_enter_graphics();
		*effect = gs_effect_create_from_file(effectPathPtr, nullptr);
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_leave_graphics();
	}

	detect_filter_update(tf, settings);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);

		gs_effect_destroy(tf->pixelateEffect);
obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

//MOD x basket
static bool areClose(const cv::Rect2f &a, const cv::Rect2f &b, float maxDist)
{
    cv::Point2f ca(a.x + a.width * 0.5f, a.y + a.height * 0.5f);
    cv::Point2f cb(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
    return cv::norm(ca - cb) < maxDist;
}

static bool buildGroupBBox(const std::vector<Object> &objects,
                           cv::Rect2f &outBox,
                           int minPeople,
                           float maxDist)
{
    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].unseenFrames > 0)
            continue;

        cv::Rect2f groupBox = objects[i].rect;
        int count = 1;

        for (size_t j = 0; j < objects.size(); ++j) {
            if (i == j || objects[j].unseenFrames > 0)
                continue;

            if (areClose(objects[i].rect, objects[j].rect, maxDist)) {
                groupBox |= objects[j].rect;
                count++;
            }
        }

        if (count >= minPeople) {
            outBox = groupBox;
            return true;
        }
    }
    return false;
}

// --- Group clustering helpers (multiple clusters) ---
// Builds connected components using the same "areClose" criterion (transitive closure).
struct GroupCluster
{
	cv::Rect2f box;
	int count = 0;
};

// Full clustering (used for preview): returns all clusters (count>=minPeople), sorted by area desc.
static std::vector<GroupCluster> buildGroupClusters(const std::vector<Object> &objects,
							    int minPeople,
							    float maxDist)
{
	std::vector<size_t> visibleIdx;
	visibleIdx.reserve(objects.size());
	for (size_t i = 0; i < objects.size(); ++i) {
		if (objects[i].unseenFrames == 0)
			visibleIdx.push_back(i);
	}

	std::vector<GroupCluster> clusters;
	if (visibleIdx.empty())
		return clusters;

	// Reuse buffer to avoid per-frame allocations (CPU spikes)
	static std::vector<uint8_t> visited;
	visited.assign(objects.size(), 0);

	std::vector<size_t> stack;
	stack.reserve(visibleIdx.size());

	for (size_t seedPos = 0; seedPos < visibleIdx.size(); ++seedPos) {
		size_t seed = visibleIdx[seedPos];
		if (visited[seed])
			continue;

		// BFS/DFS over "close" graph (transitive)
		stack.clear();
		stack.push_back(seed);
		visited[seed] = 1;

		cv::Rect2f box = objects[seed].rect;
		int count = 0;

		while (!stack.empty()) {
			size_t u = stack.back();
			stack.pop_back();
			++count;
			box |= objects[u].rect;

			// Compare only against not-yet-visited visible nodes.
			for (size_t vPos = 0; vPos < visibleIdx.size(); ++vPos) {
				size_t v = visibleIdx[vPos];
				if (visited[v])
					continue;
				if (areClose(objects[u].rect, objects[v].rect, maxDist)) {
					visited[v] = 1;
					stack.push_back(v);
				}
			}
		}

		if (count >= std::max(1, minPeople)) {
			GroupCluster c;
			c.box = box;
			c.count = count;
			clusters.push_back(c);
		}
	}

	// Prefer biggest cluster first (useful for selecting boundingBox / consistent labels)
// Basket-friendly ordering: first by people count (desc), then by area (desc).
	std::sort(clusters.begin(), clusters.end(), [](const GroupCluster &a, const GroupCluster &b) {
		if (a.count != b.count)
			return a.count > b.count;
		return a.box.area() > b.box.area();
	});

	return clusters;
}

// Best-cluster selection (used for crop/tracking):
// - avoids storing/sorting all clusters
// - early-exit when a cluster contains *all* visible people (cannot be beaten by count)
static bool selectBestGroupCluster(const std::vector<Object> &objects,
					  int minPeople,
					  float maxDist,
					  GroupCluster &bestOut)
{
	std::vector<size_t> visibleIdx;
	visibleIdx.reserve(objects.size());
	for (size_t i = 0; i < objects.size(); ++i) {
		if (objects[i].unseenFrames == 0)
			visibleIdx.push_back(i);
	}
	if (visibleIdx.empty())
		return false;

	static std::vector<uint8_t> visited;
	visited.assign(objects.size(), 0);

	std::vector<size_t> stack;
	stack.reserve(visibleIdx.size());

	bool found = false;
	float bestArea = -1.0f;

	for (size_t seedPos = 0; seedPos < visibleIdx.size(); ++seedPos) {
		const size_t seed = visibleIdx[seedPos];
		if (visited[seed])
			continue;

		stack.clear();
		stack.push_back(seed);
		visited[seed] = 1;

		cv::Rect2f box = objects[seed].rect;
		int count = 0;

		while (!stack.empty()) {
			size_t u = stack.back();
			stack.pop_back();
			++count;
			box |= objects[u].rect;

			for (size_t vPos = 0; vPos < visibleIdx.size(); ++vPos) {
				size_t v = visibleIdx[vPos];
				if (visited[v])
					continue;
				if (areClose(objects[u].rect, objects[v].rect, maxDist)) {
					visited[v] = 1;
					stack.push_back(v);
				}
			}
		}

		if (count >= std::max(1, minPeople)) {
			const float area = box.area();
			if (!found || count > bestOut.count || (count == bestOut.count && area > bestArea)) {
			bestOut.box = box;
			bestOut.count = count;
			bestArea = area;
			found = true;
		}
			// EARLY-EXIT (crop): if this cluster contains everyone visible, it's maximal by count.
			if (count == (int)visibleIdx.size()) {
				return true;
			}
		}
	}

	return found;
}

static void drawGroupClusters(cv::Mat &frame,
			      const std::vector<Object> &objects,
			      int minPeople,
			      float maxDist,
			      bool showLabel)
{
	auto clusters = buildGroupClusters(objects, minPeople, maxDist);
	if (clusters.empty())
		return;

	// Style: thicker stroke for clusters
	const int thickness = 3;
	for (size_t i = 0; i < clusters.size(); ++i) {
		const auto &c = clusters[i];
		cv::Rect r = c.box;
		// Clamp to image bounds (avoid OpenCV assertions)
		r &= cv::Rect(0, 0, frame.cols, frame.rows);
		if (r.width <= 0 || r.height <= 0)
			continue;

		cv::rectangle(frame, r, cv::Scalar(255, 0, 255), thickness);

		if (!showLabel)
			continue;

		// Label: "G1 (N)" — build string only when enabled
		const std::string label = "G" + std::to_string(i + 1) + " (" + std::to_string(c.count) + ")";
		int baseline = 0;
		auto ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
		const int x = std::max(0, r.x);
		const int y = std::max(ts.height + 4, r.y);
		cv::rectangle(frame, cv::Rect(x, y - ts.height - 4, ts.width + 6, ts.height + 6),
			      cv::Scalar(0, 0, 0), -1);
		cv::putText(frame, label, cv::Point(x + 3, y),
			    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->model) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock()) {
			// No data to process
			return;
		}
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat inferenceFrame;

	cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
	if (tf->crop_enabled) {
		cropRect = cv::Rect(tf->crop_left, tf->crop_top,
				    imageBGRA.cols - tf->crop_left - tf->crop_right,
				    imageBGRA.rows - tf->crop_top - tf->crop_bottom);
		cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
	} else {
		cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
	}

	std::vector<Object> objects;

// ---- CPU OPT: throttle inference (reuse cached objects) ----
const uint64_t nowInferNs = os_gettime_ns();
const uint64_t inferIntervalNs =
	(tf->infer_interval_ms > 0) ? (uint64_t)tf->infer_interval_ms * 1000000ULL : 0ULL;

const bool skipInfer = (inferIntervalNs > 0ULL) &&
		tf->cached_objects_valid &&
		(nowInferNs - tf->last_infer_ts_ns) < inferIntervalNs;

if (skipInfer) {
	objects = tf->cached_objects;
} else {
	// ---- CPU OPT: downscale only for inference ----
	static cv::Mat inferScaled;
	cv::Mat &inferInput = inferenceFrame;

	if (tf->infer_scale < 0.999f) {
		const double s = (double)tf->infer_scale;
		cv::resize(inferenceFrame, inferScaled, cv::Size(), s, s, cv::INTER_LINEAR);
		inferInput = inferScaled;
	}

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->model->inference(inferInput);
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	// If inference ran on a scaled frame, rescale detections back to inferenceFrame coordinates
	if (tf->infer_scale < 0.999f && tf->infer_scale > 0.0f) {
		const float inv = 1.0f / tf->infer_scale;
		for (Object &obj : objects) {
			obj.rect.x *= inv;
			obj.rect.y *= inv;
			obj.rect.width *= inv;
			obj.rect.height *= inv;
		}
	}

	// cache results
	tf->cached_objects = objects;
	tf->cached_objects_valid = true;
	tf->last_infer_ts_ns = nowInferNs;
}
if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
		}
	}

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object",
					    tf->classNames[objects[0].label].c_str());
			// release the source settings
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->minAreaThreshold > 0) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(), [tf](const Object &obj) {
				return obj.rect.area() <= (float)tf->minAreaThreshold;
			}),
			objects.end());
	}

if (tf->objectCategory != -1) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(), [tf](const Object &obj) {
				return obj.label != tf->objectCategory;
			}),
			objects.end());
	}

if (tf->sortTracking) {
		objects = tf->tracker.update(objects);
	}

	if (!tf->showUnseenObjects) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			objects.end());
	}

	if (!tf->saveDetectionsPath.empty()) {
		// Throttle disk writes a bit to reduce per-frame overhead
		static uint64_t saveCounter = 0;
		if ((++saveCounter % 5) == 0) {
			std::ofstream detectionsFile(tf->saveDetectionsPath);
			if (detectionsFile.is_open()) {
				nlohmann::json j;
				for (const Object &obj : objects) {
					nlohmann::json obj_json;
					obj_json["label"] = obj.label;
					obj_json["confidence"] = obj.prob;
					obj_json["rect"] = {{"x", obj.rect.x},
							  {"y", obj.rect.y},
							  {"width", obj.rect.width},
							  {"height", obj.rect.height}};
					obj_json["id"] = obj.id;
					j.push_back(obj_json);
				}
				// Compact JSON (no indentation) to reduce CPU + IO
				detectionsFile << j.dump();
				detectionsFile.close();
			} else {
				obs_log(LOG_ERROR, "Failed to open file for writing detections: %s",
					tf->saveDetectionsPath.c_str());
			}
		}
	}

	if (tf->preview || tf->maskingEnabled) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		if (tf->preview && tf->crop_enabled) {
			// draw the crop rectangle on the frame in a dashed line
			drawDashedRectangle(frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
		}
// Safe ROI (decision region) overlay + current decision bbox
if (tf->preview && tf->trackingEnabled && tf->trackingFilter) {
	const RectI safe = make_safe_roi(frame.cols, frame.rows,
				 tf->safe_roi_left, tf->safe_roi_right,
				 tf->safe_roi_top, tf->safe_roi_bottom);
	cv::Rect safeRect(safe.x, safe.y, safe.w, safe.h);
	drawDashedRectangle(frame, safeRect, cv::Scalar(255, 255, 0), 3, 6, 10); // yellow

	// BBox currently driving crop decision (safe / hold / fallback)
	if (rect_valid(tf->safe_roi_decision_bbox)) {
		cv::Rect decisionRect((int)tf->safe_roi_decision_bbox.x, (int)tf->safe_roi_decision_bbox.y,
				     (int)tf->safe_roi_decision_bbox.width, (int)tf->safe_roi_decision_bbox.height);
		cv::Scalar col = tf->safe_roi_decision_from_safe ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 165, 255); // magenta vs orange
		if (tf->safe_roi_decision_is_hold)
			col = cv::Scalar(255, 255, 0); // yellow for HOLD
		drawDashedRectangle(frame, decisionRect, col, 3, 6, 10);
		const char *lbl = tf->safe_roi_decision_is_hold ? "DECISION (HOLD)" :
				 (tf->safe_roi_decision_from_safe ? "DECISION (SAFE)" : "DECISION (FALLBACK)");
		cv::putText(frame, lbl,
			    cv::Point(decisionRect.x + 6, std::max(20, decisionRect.y - 8)),
			    cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2);
	}

	// Cluster inertia overlay (only meaningful in group mode)
	if (tf->zoomObject == "group") {
		if (tf->cluster_active_valid && rect_valid(tf->cluster_active_box)) {
			cv::Rect activeRect((int)tf->cluster_active_box.x, (int)tf->cluster_active_box.y,
					    (int)tf->cluster_active_box.width, (int)tf->cluster_active_box.height);
			drawDashedRectangle(frame, activeRect, cv::Scalar(0, 255, 0), 2, 6, 10); // green
			cv::putText(frame, "CLUSTER ACTIVE",
				    cv::Point(activeRect.x + 6, std::max(20, activeRect.y - 8)),
				    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
		}
		if (tf->cluster_pending_since_ns != 0 && rect_valid(tf->cluster_pending_box)) {
			cv::Rect pendingRect((int)tf->cluster_pending_box.x, (int)tf->cluster_pending_box.y,
					     (int)tf->cluster_pending_box.width, (int)tf->cluster_pending_box.height);
			drawDashedRectangle(frame, pendingRect, cv::Scalar(255, 0, 0), 2, 3, 6); // blue/red-ish
			cv::putText(frame, "CLUSTER PENDING",
				    cv::Point(pendingRect.x + 6, std::max(20, pendingRect.y - 8)),
				    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
		}
	}

	if (tf->safe_roi_holding) {
		cv::putText(frame, "SAFE ROI HOLD", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
			    1.0, cv::Scalar(255, 255, 0), 2);
	}
}
		if (tf->preview && objects.size() > 0) {
			draw_objects(frame, objects, tf->classNames);
		}
		// Optional debug overlay: draw one bbox per detected people-cluster (group).
		// This is useful when zoomObject == "group" to understand which cluster is being selected.
		if (tf->preview && tf->zoomObject == "group" && !objects.empty()) {
			int visibleCount = 0;
			for (const Object &obj : objects) {
				if (obj.unseenFrames == 0)
					visibleCount++;
			}
			int minPeople = std::max(1, tf->groupMinPeople);
			if (!tf->groupMinPeopleStrict && visibleCount > 0)
				minPeople = std::min(minPeople, visibleCount);
			const float maxDist = static_cast<float>(frame.cols) * tf->groupMaxDistFrac;
			if (tf->previewGroupClusters)
				drawGroupClusters(frame, objects, minPeople, maxDist, tf->previewGroupClusterLabel);
		}


		cv::Mat finalMask;
		if (tf->maskingEnabled) {
			finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
			for (const Object &obj : objects) {
				cv::rectangle(finalMask, obj.rect, cv::Scalar(255), -1);
			}
			if (tf->maskingDilateIterations > 0) {
				cv::dilate(finalMask, finalMask, cv::Mat(), cv::Point(-1, -1),
					   tf->maskingDilateIterations);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		if (tf->maskingEnabled) {
			finalMask.copyTo(tf->outputMask);
		}
		cv::cvtColor(frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		const int width = imageBGRA.cols;
		const int height = imageBGRA.rows;

		cv::Rect2f boundingBox = cv::Rect2f(0, 0, (float)width, (float)height);
		// Safe ROI (decision region): affects ONLY crop decision, not inference
		const RectI safe = make_safe_roi(width, height,
						 tf->safe_roi_left, tf->safe_roi_right,
						 tf->safe_roi_top, tf->safe_roi_bottom);

		std::vector<Object> safeObjects;
		safeObjects.reserve(objects.size());
		for (const Object &obj : objects) {
			if (obj.unseenFrames > 0)
				continue;
			if (!obj_center_in_safe(obj, safe))
				continue;
			safeObjects.push_back(obj);
		}

		auto compute_bbox_from_vec = [&](const std::vector<Object> &objs, bool allowCache, cv::Rect2f &outBox) -> bool {
			outBox = cv::Rect2f(0, 0, (float)width, (float)height);

			// Count visible (objs here are already visible)
			const int visibleCount = (int)objs.size();

			if (tf->zoomObject == "single") {
				if (visibleCount > 0) {
					outBox = objs.front().rect;
					return true;
				}
				return false;

			} else if (tf->zoomObject == "group") {

				cv::Rect2f groupBox;
				const float maxDist = static_cast<float>(width) * tf->groupMaxDistFrac; // configurable

				int minPeople = std::max(1, tf->groupMinPeople);
				if (!tf->groupMinPeopleStrict && visibleCount > 0)
					minPeople = std::min(minPeople, visibleCount);

				const uint64_t nowNs = os_gettime_ns();
				const uint64_t throttleNs = 100000000ULL; // 100 ms

				if (allowCache && tf->lastGroupBoxValid && (nowNs - tf->lastGroupBoxTsNs) < throttleNs) {
					outBox = tf->lastGroupBox;
					return true;
				}

				if (visibleCount > 0) {
					GroupCluster best;
					if (selectBestGroupCluster(objs, minPeople, maxDist, best)) {

						// Cluster temporal inertia: avoid micro-switching between clusters
						// Sport veloce default: 150ms
						const float switchDistPx = std::max(20.0f, 0.03f * (float)width); // ~3% of width
						tf->cluster_inertia_pending = false;

						if (!tf->cluster_active_valid) {
							tf->cluster_active_box = best.box;
							tf->cluster_active_valid = true;
							tf->cluster_pending_since_ns = 0;
							tf->cluster_pending_box = cv::Rect2f(0, 0, 0, 0);
						} else {
							const float d = rect_center_dist(best.box, tf->cluster_active_box);
							if (d <= switchDistPx) {
								// same (or very close) cluster: update immediately
								tf->cluster_active_box = best.box;
								tf->cluster_pending_since_ns = 0;
								tf->cluster_pending_box = cv::Rect2f(0, 0, 0, 0);
							} else {
								// different cluster: wait inertia before switching
								if (tf->cluster_inertia_ms <= 0) {
									tf->cluster_active_box = best.box;
									tf->cluster_pending_since_ns = 0;
									tf->cluster_pending_box = cv::Rect2f(0, 0, 0, 0);
								} else {
									if (tf->cluster_pending_since_ns == 0) {
										tf->cluster_pending_since_ns = nowNs;
										tf->cluster_pending_box = best.box;
									} else {
										const uint64_t elapsedMs = (nowNs - tf->cluster_pending_since_ns) / 1000000ULL;

										// if pending target "jumps" during waiting, restart pending (avoid flip-flop)
										const float dp = rect_center_dist(tf->cluster_pending_box, best.box);
										if (dp > switchDistPx) {
											tf->cluster_pending_since_ns = nowNs;
											tf->cluster_pending_box = best.box;
										} else if ((int)elapsedMs >= tf->cluster_inertia_ms) {
											tf->cluster_active_box = tf->cluster_pending_box;
											tf->cluster_pending_since_ns = 0;
											tf->cluster_pending_box = cv::Rect2f(0, 0, 0, 0);
										} else {
											tf->cluster_inertia_pending = true;
										}
									}
								}
							}
						}
						outBox = tf->cluster_active_box;

						if (allowCache) {
							tf->lastGroupBox = best.box;
							tf->lastGroupCount = best.count;
							tf->lastGroupBoxValid = true;
							tf->lastGroupBoxTsNs = nowNs;
						}
						return true;
					} else {
						// fallback: union of all visible objects
						bool first = true;
						for (const Object &obj : objs) {
							if (first) {
								groupBox = obj.rect;
								first = false;
							} else {
								groupBox |= obj.rect;
							}
						}
						if (!first) {
							outBox = groupBox;

							if (allowCache) {
								tf->lastGroupBox = groupBox;
								tf->lastGroupCount = visibleCount;
								tf->lastGroupBoxValid = true;
								tf->lastGroupBoxTsNs = nowNs;
							}
							return true;
						}
					}
				}

				if (allowCache)
					tf->lastGroupBoxValid = false;

				return false;

			} else if (tf->zoomObject == "biggest") {

				float maxArea = 0.0f;
				bool found = false;
				for (const Object &obj : objs) {
					float area = obj.rect.area();
					if (area > maxArea) {
						maxArea = area;
						outBox = obj.rect;
						found = true;
					}
				}
				return found;

			} else if (tf->zoomObject == "oldest") {

				uint64_t oldestId = UINT64_MAX;
				bool found = false;
				for (const Object &obj : objs) {
					if (obj.id < oldestId) {
						oldestId = obj.id;
						outBox = obj.rect;
						found = true;
					}
				}
				return found;

			} else { // all

				if (!objs.empty()) {
					outBox = objs.front().rect;
					for (size_t i = 1; i < objs.size(); ++i)
						outBox |= objs[i].rect;
					return true;
				}
				return false;
			}
		};

		const uint64_t nowNs = os_gettime_ns();

		// 1) Try SAFE decision set
		cv::Rect2f safeBox;
		const bool safeOk = (!safeObjects.empty()) && compute_bbox_from_vec(safeObjects, false, safeBox) && rect_valid(safeBox);

		if (safeOk) {
			boundingBox = safeBox;
			tf->safe_roi_last_good_bbox = safeBox;
			tf->safe_roi_decision_bbox = safeBox;
			tf->safe_roi_decision_from_safe = true;
			tf->safe_roi_decision_is_hold = false;
			tf->safe_roi_hold_until_ns = nowNs + (uint64_t)tf->safe_roi_hold_ms * 1000000ULL;
			tf->safe_roi_holding = false;
		} else if (rect_valid(tf->safe_roi_last_good_bbox) && nowNs < tf->safe_roi_hold_until_ns) {
			// 2) HOLD last safe bbox briefly
			boundingBox = tf->safe_roi_last_good_bbox;
			tf->safe_roi_holding = true;
			tf->safe_roi_decision_bbox = tf->safe_roi_last_good_bbox;
			tf->safe_roi_decision_from_safe = true;
			tf->safe_roi_decision_is_hold = true;
		} else {
			// 3) FALLBACK: compute from full visible set (existing behavior)
			std::vector<Object> visible;
			visible.reserve(objects.size());
			for (const Object &obj : objects) {
				if (obj.unseenFrames == 0)
					visible.push_back(obj);
			}

			cv::Rect2f fullBox;
			const bool fullOk = (!visible.empty()) && compute_bbox_from_vec(visible, true, fullBox) && rect_valid(fullBox);
			if (fullOk)
				boundingBox = fullBox;
			// debug decision
			tf->safe_roi_decision_bbox = fullOk ? fullBox : cv::Rect2f(0,0,0,0);
			tf->safe_roi_decision_from_safe = false;
			tf->safe_roi_decision_is_hold = false;

			tf->safe_roi_holding = false;
		}

		bool lostTracking = true;
		for (const Object &obj : objects) {
			if (obj.unseenFrames == 0) {
				lostTracking = false;
				break;
			}
		}
                // the zooming box should maintain the aspect ratio of the image
                // with the tf->zoomFactor controlling the effective buffer around the bounding box
                // the bounding box is the center of the zooming box

// Maintain the aspect ratio of the image.
// Default behaviour: dynamic zoom box (old behaviour).
// For "group": horizontal pan only (no zoom), full height, fixed window width.
float frameAspectRatio = (float)width / (float)height;

float zx = 0.0f, zy = 0.0f, zw = 0.0f, zh = 0.0f;

if (tf->zoomObject == "group") {
        // Pan-only on X: keep full height (no vertical crop) and keep a fixed window width.
        // Reuse zoomFactor as "horizontal coverage" (0..1], where 1 means full width.
        float coverage = tf->zoomFactor;
        if (coverage <= 0.0f) coverage = 1.0f;
        if (coverage > 1.0f)  coverage = 1.0f;

        zh = (float)height;
        zw = (float)width * coverage;

        // Safety clamp: window cannot exceed frame.
        if (zw > (float)width) zw = (float)width;
        if (zw < 1.0f) zw = 1.0f;

        // Choose target X based on preset (manual left/center/right or auto-follow group).
        float maxZX = (float)width - zw;
        if (maxZX < 0.0f) maxZX = 0.0f;

        if (tf->x_pan_preset == "left") {
                zx = 0.0f;
        } else if (tf->x_pan_preset == "right") {
                zx = maxZX;
        } else if (tf->x_pan_preset == "center") {
                zx = maxZX * 0.5f;
	        } else if (tf->x_pan_preset == "autosnap" || tf->x_pan_preset == "autosnap_smooth") {
	                // Auto-snap between Left/Center/Right with hysteresis to avoid micro-movements.
                float targetCenterX = boundingBox.x + (boundingBox.width * 0.5f);
                float norm = (width > 0) ? (targetCenterX / (float)width) : 0.5f;
                float h = tf->x_snap_hysteresis;
                if (h < 0.0f) h = 0.0f;
                if (h > 0.20f) h = 0.20f;
                const float t1 = 1.0f / 3.0f;
                const float t2 = 2.0f / 3.0f;

	                // Update snap state with hysteresis.
	                const int prev_state = tf->x_snap_state;
	                switch (tf->x_snap_state) {
                case 0: // left
                        if (norm > t1 + h) tf->x_snap_state = 1;
                        break;
                case 2: // right
                        if (norm < t2 - h) tf->x_snap_state = 1;
                        break;
                case 1: // center
                default:
                        if (norm < t1 - h) tf->x_snap_state = 0;
                        else if (norm > t2 + h) tf->x_snap_state = 2;
                        break;
                }

	                // Target position for this snap state.
	                float targetZX = (tf->x_snap_state == 0) ? 0.0f : (tf->x_snap_state == 2) ? maxZX : (maxZX * 0.5f);
	                zx = targetZX;

	                if (tf->x_pan_preset == "autosnap") {
	                        // Hard snap: kill residual velocity every frame so it truly "snaps".
	                        tf->trackVelX = 0.0f;
	                } else {
	                        // Smooth snap: only reset velocity when we change lane, to get a clean 200-300ms move.
	                        if (tf->x_snap_state != prev_state)
	                                tf->trackVelX = 0.0f;
	                }
        } else {
                // Auto: center the window on the group's center X.
                float targetCenterX = boundingBox.x + (boundingBox.width * 0.5f);
                zx = targetCenterX - (zw * 0.5f);
        }
        zy = 0.0f;

        // End-stops (fine corsa): clamp within [0, width-zw]
        if (zx < 0.0f) zx = 0.0f;
        if (zx > maxZX) zx = maxZX;
} else {
        // calculate an aspect ratio box around the object using its height
        float boxHeight = boundingBox.height;
        // calculate the zooming box size
        float dh = (float)height - boxHeight;
        float buffer = dh * (1.0f - tf->zoomFactor);
        zh = boxHeight + buffer;
        zw = zh * frameAspectRatio;
        // calculate the top left corner of the zooming box
        zx = boundingBox.x - (zw - boundingBox.width) / 2.0f;
        zy = boundingBox.y - (zh - boundingBox.height) / 2.0f;
}
				// --- Optional X deadband (applies to ALL zoom_object modes) ---
			if (tf->x_deadband > 0.0f) {
				const float db_px = (tf->x_deadband / 100.0f) * (float)width;

				if (tf->has_last_target_zx) {
				const float dx = zx - tf->last_target_zx;
					if (std::fabs(dx) < db_px) {
						zx = tf->last_target_zx; // ignore micro jitter
					} else {
					tf->last_target_zx = zx;
					}
			} else {
					tf->last_target_zx = zx;
					tf->has_last_target_zx = true;
			}
}
                if (tf->trackingRect.width == 0) {
                        // initialize the trackingRect
                        tf->trackingRect = cv::Rect2f(zx, zy, zw, zh);
				tf->trackVelX = tf->trackVelY = tf->trackVelW = tf->trackVelH = 0.0f;
                } else {
                        // interpolate the zooming box to tf->trackingRect (frame-rate independent, low hysteresis)
                        const float alpha60 = tf->zoomSpeedFactor * (lostTracking ? 0.2f : 1.0f);

                        if (alpha60 <= 0.0f) {
                        	// frozen
                        } else if (alpha60 >= 1.0f) {
                        	// snap
                        	tf->trackingRect = cv::Rect2f(zx, zy, zw, zh);
				tf->trackVelX = tf->trackVelY = tf->trackVelW = tf->trackVelH = 0.0f;
                        	tf->trackVelX = tf->trackVelY = tf->trackVelW = tf->trackVelH = 0.0f;
                        } else {

	const float smoothTime = smooth_time_from_alpha60(alpha60);

	if (tf->zoomObject == "group") {
		// Pan-only X: smooth only the horizontal movement.
			float smoothTimeX = smoothTime;
			// Auto-snap (smooth): use an explicit transition time so moves take ~200-300ms regardless of FPS.
			if (tf->x_pan_preset == "autosnap_smooth") {
				smoothTimeX = std::clamp(tf->x_snap_transition_time, 0.05f, 1.0f);
			}
			tf->trackingRect.x = smooth_damp_critically_damped(tf->trackingRect.x, zx, tf->trackVelX,
							   smoothTimeX, seconds);

		// Keep fixed size and full height; no vertical movement.
		tf->trackingRect.y = zy;
		tf->trackingRect.width = zw;
		tf->trackingRect.height = zh;

		// Avoid accumulating velocities on unused axes.
		tf->trackVelY = 0.0f;
		tf->trackVelW = 0.0f;
		tf->trackVelH = 0.0f;
	} else {
		tf->trackingRect.x = smooth_damp_critically_damped(tf->trackingRect.x, zx, tf->trackVelX,
							   smoothTime, seconds);
		tf->trackingRect.y = smooth_damp_critically_damped(tf->trackingRect.y, zy, tf->trackVelY,
							   smoothTime, seconds);
		tf->trackingRect.width =
			smooth_damp_critically_damped(tf->trackingRect.width, zw, tf->trackVelW, smoothTime, seconds);
		tf->trackingRect.height =
			smooth_damp_critically_damped(tf->trackingRect.height, zh, tf->trackVelH, smoothTime, seconds);
	}
}
}

                // get the settings of the crop/pad filter
			obs_data_t *crop_pad_settings = obs_source_get_settings(tf->trackingFilter);

			// Clamp to valid crop values to avoid negative crop/pad inputs
			const float x0 = std::max(0.0f, tf->trackingRect.x);
			const float y0 = std::max(0.0f, tf->trackingRect.y);
			const float x1 = std::min((float)width, tf->trackingRect.x + tf->trackingRect.width);
			const float y1 = std::min((float)height, tf->trackingRect.y + tf->trackingRect.height);

			const int left = (int)x0;
			const int top = (int)y0;
			const int right = (int)((float)width - x1);
			const int bottom = (int)((float)height - y1);

			obs_data_set_int(crop_pad_settings, "left", left);
			obs_data_set_int(crop_pad_settings, "top", top);
			obs_data_set_int(crop_pad_settings, "right", std::max(0, right));
			obs_data_set_int(crop_pad_settings, "bottom", std::max(0, bottom));

			// apply the settings
                obs_source_update(tf->trackingFilter, crop_pad_settings);
                obs_data_release(crop_pad_settings);
        }
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->model) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview is enabled, render the image
	if (tf->preview || tf->maskingEnabled) {
		cv::Mat outputBGRA, outputMask;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
			outputMask = tf->outputMask.clone();
		}

		gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
						      (const uint8_t **)&outputBGRA.data, 0);
		gs_texture_t *maskTexture = nullptr;
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		if (tf->maskingEnabled) {
			maskTexture = gs_texture_create(width, height, GS_R8, 1,
							(const uint8_t **)&outputMask.data, 0);
			gs_effect_set_texture(maskParam, maskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				gs_texture_destroy(tex);
				tex = blur_image(tf, width, height, maskTexture);
			} else if (tf->maskingType == "pixelate") {
				gs_texture_destroy(tex);
				tex = pixelate_image(tf, width, height, maskTexture,
						     (float)tf->maskingBlurRadius);
			} else if (tf->maskingType == "transparent") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, 0);
			} else if (tf->maskingType == "solid_color") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, tf->maskingColor);
			}
		}

		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_texture_destroy(tex);
		gs_texture_destroy(maskTexture);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}