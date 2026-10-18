#include "sequence_utils.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

SequenceFeatures extractSequenceFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                         const std::vector<cv::Rect>& bbox_sequence) {

    SequenceFeatures features;
    if (mask_sequence.size() < 2 || mask_sequence.size() != bbox_sequence.size()) {
        return features;
    }

    features.net_displacement = computeNetDisplacement(bbox_sequence);
    features.avg_velocity = computeAverageSequenceVelocity(bbox_sequence);
    features.stride_peaks = computeStrideFrequency_Width(bbox_sequence);

    return features;
}

StaticFeatures computeStaticFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                     const std::vector<cv::Rect>& bbox_sequence) {

    StaticFeatures feats;
    if (mask_sequence.size() < 2 || bbox_sequence.empty()) {
        return feats;
    }

    std::vector<double> cy_values;        // Stores normalized vertical motion centroids (0.0 = top, 1.0 = bottom) for valid frames
    std::vector<double> x_offsets;        // Stores normalized lateral deviations from the bounding box midline (|cx - mid| / width)
    const size_t MIN_ACTIVE_PIXELS = 30; // Noise rejection threshold to skip nearly motionless frames

    for (size_t i = 1; i < mask_sequence.size(); ++i) {
        cv::Rect bbox = bbox_sequence[i];
        if (bbox.width <= 0 || bbox.height <= 0) continue;

        // Extract frame-to-frame motion mask
        cv::Mat diff;
        cv::absdiff(mask_sequence[i], mask_sequence[i - 1], diff);

        // Clip the bounding box to ensure it stays strictly within image boundaries
        cv::Rect img_bounds(0, 0, diff.cols, diff.rows);
        cv::Rect valid_roi = bbox & img_bounds;
        if (valid_roi.area() <= 0) continue;

        cv::Mat roi_diff = diff(valid_roi);

        // Collect coordinates of all pixels exhibiting temporal change
        std::vector<cv::Point> non_zero_pts;
        cv::findNonZero(roi_diff, non_zero_pts);

        // Discard frames with insufficient motion to prevent centroid noise
        if (non_zero_pts.size() >= MIN_ACTIVE_PIXELS) {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for (const auto& pt : non_zero_pts) {
                sum_x += pt.x;
                sum_y += pt.y;
            }

            // Local motion center of mass within the bounding box
            double cx = sum_x / non_zero_pts.size();
            double cy = sum_y / non_zero_pts.size();

            // Normalize vertical reach by bbox height (0.0 = top of head, 1.0 = feet)
            double norm_cy = cy / bbox.height;
            cy_values.push_back(norm_cy);

            // Compute normalized lateral offset relative to the bbox's midline
            double center_x = bbox.width / 2.0;
            double norm_x_offset = std::abs(cx - center_x) / bbox.width;
            x_offsets.push_back(norm_x_offset);
        }
    }

    // Case where the sequence had no frames with sufficient motion
    if (cy_values.empty()) {
        return feats;
    }

    // Determine highest and lowest vertical reach across the entire action
    feats.min_cy = 1.0;
    feats.max_cy = 0.0;
    for (double cy : cy_values) {
        if (cy < feats.min_cy) feats.min_cy = cy;
        if (cy > feats.max_cy) feats.max_cy = cy;
    }
    feats.y_range = feats.max_cy - feats.min_cy;

    // Compute mean lateral asymmetry over all active frames
    double sum_x_offset = 0.0;
    for (double offset : x_offsets) {
        sum_x_offset += offset;
    }
    feats.avg_x_offset = sum_x_offset / x_offsets.size();
    feats.is_valid = true;

    return feats;
}

std::vector<cv::Mat> overlayActionOnFrames(const std::vector<cv::Mat>& frames, const std::string& action_name) {
    std::vector<cv::Mat> annotated_frames;
    annotated_frames.reserve(frames.size());

    const std::string text = "Action: " + action_name;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    
    const double font_scale = 0.3;
    const int thickness = 1;
    const cv::Scalar text_color(0, 0, 255); // Red in BGR format

    for (const auto& original_frame : frames) {
        if (original_frame.empty()) {
            annotated_frames.push_back(cv::Mat());
            continue;
        }

        cv::Mat display_frame;
        // Convert single-channel grayscale images to 3-channel BGR to allow color rendering
        if (original_frame.channels() == 1) {
            cv::cvtColor(original_frame, display_frame, cv::COLOR_GRAY2BGR);
        } else {
            display_frame = original_frame.clone();
        }

        // Anchor origin: 4 pixels from the left border, 6 pixels above the bottom edge
        cv::Point text_origin(4, display_frame.rows - 6);

        // Render the red text onto the cloned frame
        cv::putText(display_frame, text, text_origin, font_face, font_scale, 
                    text_color, thickness, cv::LINE_AA);

        annotated_frames.push_back(display_frame);
    }

    return annotated_frames;
}