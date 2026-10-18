#ifndef SEQUENCE_UTILS_H
#define SEQUENCE_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "bbox_utils.h"


/**
 * @brief Kinematic descriptors extracted from bounding box trajectories.
 */
struct SequenceFeatures {
    float net_displacement = 0.0f;
    float avg_velocity = 0.0f;
    int stride_peaks = 0;
};

/**
 * @brief Spatial and centroid descriptors extracted from frame-by-frame difference masks.
 */
struct StaticFeatures {
    double min_cy = 1.0;       // Highest vertical reach (0.0 = top of bbox, 1.0 = bottom)
    double max_cy = 0.0;       // Lowest vertical reach
    double y_range = 0.0;      // Total vertical excursion (max_cy - min_cy)
    double avg_x_offset = 0.0; // Mean lateral asymmetry (|cx - center_x| / bbox.width)
    bool is_valid = false;     // True if sufficient motion was detected
};


/**
 * @brief Extracts global kinematic features from bounding box trajectories.
 * @param mask_sequence Vector of binary foreground masks across the sequence (used for size validation).
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return SequenceFeatures Structure containing net displacement, average velocity, and stride peaks.
 */
SequenceFeatures extractSequenceFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                         const std::vector<cv::Rect>& bbox_sequence);


/**
 * @brief Computes centroid trajectory and spatial motion features for static sequences.
 * @param mask_sequence Vector of binary foreground/motion masks across the sequence frames.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return StaticFeatures Structure containing vertical reach (min_cy, max_cy, y_range),
 *                        average horizontal asymmetry (avg_x_offset), and motion validity flag.
 */
StaticFeatures computeStaticFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                     const std::vector<cv::Rect>& bbox_sequence);


/**
 * @brief Creates annotated copies of the input frames with the predicted action label.
 *        Generates a cloned sequence where each frame has red text (the predicted action name) in the bottom-left corner.
 * @param frames Constant reference to the input vector of frames (the ones with a bbox).
 * @param action_name String representing the predicted action label (e.g., "boxing").
 * @return std::vector<cv::Mat> Vector containing the new annotated frame clones.
 */
std::vector<cv::Mat> overlayActionOnFrames(const std::vector<cv::Mat>& frames, const std::string& action_name);


#endif // SEQUENCE_UTILS_H