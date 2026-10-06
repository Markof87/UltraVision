#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "segm+bbox_v3.h"

namespace fs = std::filesystem;

// ============================================================================
// 1. DATA STRUCTURES
// ============================================================================

struct GroundTruthData {
    int class_id = -1;
    double xc = 0.0;
    double yc = 0.0;
    double w  = 0.0;
    double h  = 0.0;
    bool valid = false;
};

struct SequenceData {
    std::string action_name;
    std::string seq_name;
    std::vector<std::string> image_files;
};

struct SequenceFeatures {
    float net_displacement = 0.0f;
    float avg_velocity = 0.0f;
    int stride_peaks = 0;
};

struct StaticFeatures {
    double min_cy = 1.0;       // Highest vertical reach (0.0 = top of bbox, 1.0 = bottom)
    double max_cy = 0.0;       // Lowest vertical reach
    double y_range = 0.0;      // Total vertical excursion (max_cy - min_cy)
    double avg_x_offset = 0.0; // Mean lateral asymmetry (|cx - center_x| / bbox.width)
    bool is_valid = false;     // True if sufficient motion was detected
};

// ============================================================================
// 2. UTILITY & GROUND TRUTH FUNCTIONS
// ============================================================================

int getActionClassId(const std::string& action_name) {
    if (action_name == "boxing")        return 1;
    if (action_name == "handclapping")  return 2;
    if (action_name == "handwaving")    return 3;
    if (action_name == "jogging")       return 4;
    if (action_name == "running")       return 5;
    if (action_name == "walking")       return 6;
    return -1;
}

GroundTruthData loadGroundTruthData(const std::string& filepath) {
    GroundTruthData gt;
    std::ifstream file(filepath);
    if (!file.is_open()) return gt;

    if (file >> gt.class_id >> gt.xc >> gt.yc >> gt.w >> gt.h) {
        gt.valid = true;
    }
    return gt;
}

inline std::vector<SequenceData> loadSequences(const std::string& main_folder) {
    std::vector<SequenceData> all_sequences;

    if (!fs::exists(main_folder) || !fs::is_directory(main_folder)) {
        std::cerr << "[ERROR] Main folder does not exist: " << main_folder << std::endl;
        return all_sequences;
    }

    std::cout << "[INFO] Scansione delle sequenze del dataset KTH..." << std::endl;

    for (const auto& action_folder : fs::directory_iterator(main_folder)) {
        if (!action_folder.is_directory()) continue;
        for (const auto& seq_folder : fs::directory_iterator(action_folder.path())) {
            if (!seq_folder.is_directory()) continue;

            std::string data_path = seq_folder.path().string() + "/data/";
            if (!fs::exists(data_path)) continue;

            std::vector<std::string> image_files;
            for (const auto& entry : fs::directory_iterator(data_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".png" || ext == ".jpg") {
                        image_files.push_back(entry.path().string());
                    }
                }
            }
            std::sort(image_files.begin(), image_files.end());

            if (!image_files.empty()) {
                SequenceData seq;
                seq.action_name = action_folder.path().filename().string();
                seq.seq_name = seq_folder.path().filename().string();
                seq.image_files = image_files;
                all_sequences.push_back(seq);
            }
        }
    }

    std::cout << "[INFO] Trovate " << all_sequences.size() << " sequenze in totale." << std::endl;
    return all_sequences;
}

// ============================================================================
// 3. SEGMENTATION & BBOX DETECTION
// ============================================================================

/*
void segmentation(const cv::Mat& input, cv::Mat& output) {                        
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = input;
    }

    cv::Mat blurred_frame;
    cv::GaussianBlur(gray, blurred_frame, cv::Size(95, 95), 0, 0);

    cv::Mat grayFloat, blurredFloat, correctedFloat;
    gray.convertTo(grayFloat, CV_32FC1);
    blurred_frame.convertTo(blurredFloat, CV_32FC1);

    double mean_background = cv::mean(gray)[0];

    cv::divide(grayFloat, blurredFloat, correctedFloat);
    correctedFloat = correctedFloat * mean_background;

    cv::Mat final_gray;
    correctedFloat.convertTo(final_gray, CV_8UC1);

    cv::Mat blurred;
    cv::GaussianBlur(final_gray, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::Mat closed;
    cv::Mat closing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(27, 25));     
    cv::morphologyEx(blurred, closed, cv::MORPH_CLOSE, closing_se);

    cv::Mat subtracted;
    cv::subtract(closed, blurred, subtracted);

    cv::Mat otsu_sub;
    cv::threshold(subtracted, otsu_sub, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat almost_segmented;
    cv::Mat v_closing_se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 17));
    cv::morphologyEx(otsu_sub, almost_segmented, cv::MORPH_CLOSE, v_closing_se);

    cv::Mat segmented;
    cv::Mat smoothing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(almost_segmented, segmented, cv::MORPH_CLOSE, smoothing_se);

    output = segmented;
}

cv::Rect bbox_func4(const cv::Mat& input, cv::Mat& output) { 
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(input, labels, stats, centroids, 8, CV_32S);

    cv::Rect actor_bbox(0, 0, 0, 0);
    cv::Mat finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);

    bool actor_found = false; 
    int largest_label = -1;
    int max_area = 0;
    int min_area = 30;

    for (int j = 1; j < num_labels; j++) {
        int area = stats.at<int>(j, cv::CC_STAT_AREA);
        int w = stats.at<int>(j, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(j, cv::CC_STAT_HEIGHT);

        double bbox_area = static_cast<double>(w * h);
        double areas_ratio = static_cast<double>(area) / bbox_area;

        if (areas_ratio >= 0.20) {
            if (area > min_area && area > max_area) {
                max_area = area;
                largest_label = j;
            }
        }
    }

    if (largest_label != -1) {
        int x = stats.at<int>(largest_label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(largest_label, cv::CC_STAT_TOP);
        int w = stats.at<int>(largest_label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(largest_label, cv::CC_STAT_HEIGHT);
        actor_bbox = cv::Rect(x, y, w, h);

        finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);

        for (int r = 0; r < input.rows; r++) {
            for (int c = 0; c < input.cols; c++) {
                if (labels.at<int>(r, c) == largest_label) {
                    finalSilhouette.at<uchar>(r, c) = 255;
                }
            }
        }

        bool aspect_ratio = (h >= 12 && (h * 1.5) > w);
        int frame_area = input.rows * input.cols;
        bool big_enough = (max_area > 100 && max_area < frame_area * 0.4);

        if (aspect_ratio && big_enough) {
            actor_found = true;
        } else {
            finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);
        }
    }

    if (actor_found) {
        cv::rectangle(output, actor_bbox, cv::Scalar(0, 0, 255), 2);
        return actor_bbox;
    }
    return cv::Rect(0, 0, 0, 0);
}
*/

// ============================================================================
// 4. KINEMATIC & GEOMETRIC FEATURE EXTRACTION
// ============================================================================

inline cv::Point2f bbox_centroid(const cv::Rect& bbox) {
    if (bbox.width <= 0 || bbox.height <= 0) {
        return cv::Point2f(0.0f, 0.0f);
    }
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    return cv::Point2f(cx, cy);
}

inline float computeNetDisplacement(const std::vector<cv::Rect>& bbox_sequence) {
    std::vector<cv::Point2f> valid_centroids;
    for (const auto& bbox : bbox_sequence) {
        if (bbox.width > 0 && bbox.height > 0) {
            valid_centroids.push_back(bbox_centroid(bbox));
        }
    }

    if (valid_centroids.size() < 2) return 0.0f;

    cv::Point2f first_c = valid_centroids.front();
    cv::Point2f last_c = valid_centroids.back();
    float dx = last_c.x - first_c.x;
    float dy = last_c.y - first_c.y;

    return std::sqrt(dx * dx + dy * dy);
}

inline float computeAverageSequenceVelocity(const std::vector<cv::Rect>& bbox_sequence) {
    float total_velocity = 0.0f;
    int valid_transitions = 0;
    
    cv::Point2f prev_centroid(0.0f, 0.0f);
    bool has_prev = false;

    for (const auto& bbox : bbox_sequence) {
        if (bbox.width > 0 && bbox.height > 0) {
            cv::Point2f curr_centroid = bbox_centroid(bbox);
            
            if (has_prev) {
                float dx = curr_centroid.x - prev_centroid.x;
                float dy = curr_centroid.y - prev_centroid.y;
                float v = std::sqrt(dx * dx + dy * dy);
                
                total_velocity += v;
                valid_transitions++;
            }
            
            prev_centroid = curr_centroid;
            has_prev = true;
        } else {
            has_prev = false;
        }
    }

    if (valid_transitions == 0) return 0.0f;
    return total_velocity / valid_transitions;
}

inline int computeStrideFrequency_Width(const std::vector<cv::Rect>& bbox_sequence) {
    std::vector<float> widths;
    for (const auto& bbox : bbox_sequence) {
        if (bbox.width > 0) {
            widths.push_back(static_cast<float>(bbox.width));
        }
    }

    if (widths.size() < 3) return 0;

    std::vector<float> smoothed_widths(widths.size());
    for (size_t i = 0; i < widths.size(); ++i) {
        float sum = widths[i];
        int count = 1;
        
        if (i > 0) { 
            sum += widths[i - 1]; 
            count++; 
        }
        if (i + 1 < widths.size()) { 
            sum += widths[i + 1]; 
            count++; 
        }
        smoothed_widths[i] = sum / count;
    }

    int peaks = 0;
    for (size_t i = 1; i + 1 < smoothed_widths.size(); ++i) {
        if (smoothed_widths[i] > smoothed_widths[i - 1] && smoothed_widths[i] > smoothed_widths[i + 1]) {
            peaks++;
        }
    }
    return peaks;
}

inline SequenceFeatures extractSequenceFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                                const std::vector<cv::Rect>& bbox_sequence) 
{
    SequenceFeatures features;
    if (mask_sequence.size() < 2 || mask_sequence.size() != bbox_sequence.size()) {
        return features;
    }

    features.net_displacement = computeNetDisplacement(bbox_sequence);
    features.avg_velocity = computeAverageSequenceVelocity(bbox_sequence);
    features.stride_peaks = computeStrideFrequency_Width(bbox_sequence);

    return features;
}

inline StaticFeatures computeStaticFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                            const std::vector<cv::Rect>& bbox_sequence) 
{
    StaticFeatures feats;
    if (mask_sequence.size() < 2 || bbox_sequence.empty()) {
        return feats;
    }

    std::vector<double> cy_values;
    std::vector<double> x_offsets;
    const size_t MIN_ACTIVE_PIXELS = 30;

    for (size_t i = 1; i < mask_sequence.size(); ++i) {
        cv::Rect bbox = bbox_sequence[i];
        if (bbox.width <= 0 || bbox.height <= 0) continue;

        cv::Mat diff;
        cv::absdiff(mask_sequence[i], mask_sequence[i - 1], diff);

        cv::Rect img_bounds(0, 0, diff.cols, diff.rows);
        cv::Rect valid_roi = bbox & img_bounds;
        if (valid_roi.area() <= 0) continue;

        cv::Mat roi_diff = diff(valid_roi);

        std::vector<cv::Point> non_zero_pts;
        cv::findNonZero(roi_diff, non_zero_pts);

        if (non_zero_pts.size() >= MIN_ACTIVE_PIXELS) {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for (const auto& pt : non_zero_pts) {
                sum_x += pt.x;
                sum_y += pt.y;
            }

            double cx = sum_x / non_zero_pts.size();
            double cy = sum_y / non_zero_pts.size();

            double norm_cy = cy / bbox.height;
            cy_values.push_back(norm_cy);

            double center_x = bbox.width / 2.0;
            double norm_x_offset = std::abs(cx - center_x) / bbox.width;
            x_offsets.push_back(norm_x_offset);
        }
    }

    if (cy_values.empty()) {
        return feats;
    }

    feats.min_cy = 1.0;
    feats.max_cy = 0.0;
    for (double cy : cy_values) {
        if (cy < feats.min_cy) feats.min_cy = cy;
        if (cy > feats.max_cy) feats.max_cy = cy;
    }
    feats.y_range = feats.max_cy - feats.min_cy;

    double sum_x_offset = 0.0;
    for (double offset : x_offsets) {
        sum_x_offset += offset;
    }
    feats.avg_x_offset = sum_x_offset / x_offsets.size();
    feats.is_valid = true;

    return feats;
}

// ============================================================================
// 5. CLASSIFICATION LOGIC
// ============================================================================

std::string classifyDynamicAction_v4(const SequenceFeatures& feats) {
    if (feats.avg_velocity > 5.8f || 
       (feats.avg_velocity > 3.5f && feats.stride_peaks <= 3 && feats.net_displacement < 140.0f)) 
    {
        return "running";
    }
    
    if ((feats.avg_velocity > 2.4f && feats.stride_peaks >= 4) || feats.avg_velocity > 3.2f) {
        return "jogging";
    }
    
    return "walking";
}

std::string classifyStaticAction_v4(const StaticFeatures& feats) {
    if (feats.avg_x_offset > 0.078 || 
       (feats.avg_x_offset > 0.0665 && feats.y_range > 0.410)) 
    {
        return "boxing";
    }
    else if ((feats.min_cy < 0.185 && feats.avg_x_offset <= 0.055) ||
             (feats.min_cy < 0.285 && feats.y_range > 0.310 && feats.avg_x_offset < 0.050)) 
    {
        return "handwaving";
    }
    else {
        return "handclapping";
    }
}

inline std::string classifySequence(const SequenceFeatures& dyn_feats,
                                    const std::vector<cv::Mat>& mask_sequence,
                                    const std::vector<cv::Rect>& bbox_sequence) 
{
    bool is_dynamic = (dyn_feats.net_displacement > 25.0f || dyn_feats.avg_velocity > 4.0f);

    if (is_dynamic) {
        return classifyDynamicAction_v4(dyn_feats);
    } else {
        StaticFeatures static_feats = computeStaticFeatures(mask_sequence, bbox_sequence);
        return classifyStaticAction_v4(static_feats);
    }
}

// ============================================================================
// 6. MAIN BENCHMARK EXECUTION
// ============================================================================

int main() {
    std::string main_folder = "../Sequences";
    const size_t target_frame_idx = 19; // 20th frame (0-based index: 19)

    std::vector<SequenceData> all_sequences = loadSequences(main_folder);
    if (all_sequences.empty()) {
        std::cerr << "[ERROR] No sequences found in: " << main_folder << std::endl;
        return -1;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                 FRAME 20: PREDICTED VS GROUND TRUTH COMPARISON                 " << std::endl;
    std::cout << " Format: <class_id> <x_center> <y_center> <width> <height>                      " << std::endl;
    std::cout << "================================================================================\n" << std::endl;

    for (size_t i = 0; i < all_sequences.size(); ++i) {
        const auto& current_seq = all_sequences[i];

        std::vector<cv::Mat> mask_sequence;
        std::vector<cv::Rect> bbox_sequence;
        cv::Rect target_bbox;
        bool frame_20_found = false;

        for (size_t f = 0; f < current_seq.image_files.size(); ++f) {
            cv::Mat img = cv::imread(current_seq.image_files[f], cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                cv::Mat mask;
                segmentation(img, mask);
                cv::Rect bbox = bbox_func4(mask, img);

                mask_sequence.push_back(mask);
                bbox_sequence.push_back(bbox);

                if (f == target_frame_idx) {
                    target_bbox = bbox;
                    frame_20_found = true;
                }
            }
        }

        if (mask_sequence.empty() || !frame_20_found) {
            std::cerr << "[SKIP] Missing frame 20 or sequence empty: " << current_seq.seq_name << std::endl;
            continue;
        }

        SequenceFeatures dyn_feats = extractSequenceFeatures(mask_sequence, bbox_sequence);
        std::string pred_label = classifySequence(dyn_feats, mask_sequence, bbox_sequence);
        int pred_class_id = getActionClassId(pred_label);

        double pred_xc = target_bbox.x + (target_bbox.width / 2.0);
        double pred_yc = target_bbox.y + (target_bbox.height / 2.0);
        double pred_w  = static_cast<double>(target_bbox.width);
        double pred_h  = static_cast<double>(target_bbox.height);

        std::string gt_path = main_folder + "/" + current_seq.action_name + "/" + current_seq.seq_name + "/labels/ground_truth.txt";
        GroundTruthData gt = loadGroundTruthData(gt_path);

        std::cout << "[" << std::setw(2) << (i + 1) << "/72] " 
                  << current_seq.action_name << "/" << current_seq.seq_name << std::endl;

        std::cout << "    PRED: " 
                  << std::setw(8) << pred_class_id
                  << std::fixed << std::setprecision(1)
                  << std::setw(9) << pred_xc
                  << std::setw(9) << pred_yc
                  << std::setw(9) << pred_w
                  << std::setw(9) << pred_h << std::endl;

        if (gt.valid) {
            std::cout << "    GT:   " 
                      << std::setw(8) << gt.class_id
                      << std::fixed << std::setprecision(1)
                      << std::setw(9) << gt.xc
                      << std::setw(9) << gt.yc
                      << std::setw(9) << gt.w
                      << std::setw(9) << gt.h << std::endl;

            double diff_xc = pred_xc - gt.xc;
            double diff_yc = pred_yc - gt.yc;
            double diff_w  = pred_w  - gt.w;
            double diff_h  = pred_h  - gt.h;

            std::string class_status = (pred_class_id == gt.class_id) ? "MATCH" : "DIFF";

            std::cout << "    DIFF: " 
                      << std::setw(8) << class_status
                      << std::showpos << std::fixed << std::setprecision(1)
                      << std::setw(9) << diff_xc
                      << std::setw(9) << diff_yc
                      << std::setw(9) << diff_w
                      << std::setw(9) << diff_h
                      << std::noshowpos << "\n" << std::endl;
        } else {
            std::cout << "    GT:   [MISSING GROUND TRUTH FILE]\n" << std::endl;
        }
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "[INFO] Evaluation complete." << std::endl;

    return 0;
}