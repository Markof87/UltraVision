#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>


/**
 * @brief Computes the centroid of a bounding box.
 * @param bbox The bounding box (cv::Rect).
 * @return cv::Point2f Centroid coordinates (x, y).
 */
inline cv::Point2f bbox_centroid(const cv::Rect& bbox) {
    if (bbox.width <= 0 || bbox.height <= 0) {
        return cv::Point2f(0.0f, 0.0f);
    }
    
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    
    return cv::Point2f(cx, cy);
}


/**
 * @brief Computes the net displacement (Euclidean distance) between the first and last valid centroid of a sequence.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return float Net displacement in pixels.
 */
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


/**
 * @brief Computes the average centroid velocity for a sequence of bounding boxes.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return float Average velocity in pixels/frame.
 */
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
    return total_velocity / valid_transitions;  // The average velocity given the number of valid bboxes
}


/**
 * @brief Computes stride frequency by analyzing the width variations of the bounding box.
 *        Applies a 3-frame moving average filter to smooth out mask noise before peak detection.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return int Number of detected peaks (stride cycles).
 */
inline int computeStrideFrequency_Width(const std::vector<cv::Rect>& bbox_sequence) {
    std::vector<float> widths;
    for (const auto& bbox : bbox_sequence) {
        if (bbox.width > 0) {
            widths.push_back(static_cast<float>(bbox.width));
        }
    }

    if (widths.size() < 3) return 0;

    // Apply a 3-frame moving average filter to smooth the width signal
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
        // Detection of local maxima (bbox expansion peaks) on the smoothed signal
        if (smoothed_widths[i] > smoothed_widths[i - 1] && smoothed_widths[i] > smoothed_widths[i + 1]) {
            peaks++;
        }
    }
    return peaks;
}