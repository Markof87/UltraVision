#ifndef SEQUENCE_UTILS_H
#define SEQUENCE_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "bbox_utils.h"

struct SequenceData {
    std::string action_name;
    std::string seq_name;
    std::vector<std::string> image_files;
};

// Error logging record
struct ErrorRecord {
    std::string real_action;
    std::string predicted_action;
    std::string seq_identifier;
};

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
 * @brief Scans the KTH dataset folder and indexes all action sequences.
 * @param main_folder Path to the root dataset directory.
 * @return std::vector<SequenceData> Indexed action sequences.
 */
std::vector<SequenceData> loadSequences(const std::string& main_folder);

/**
 * @brief Interactive visualizer for checking frames, motion difference masks, and prediction status.
 * @param sequence_frames Vector of original grayscale frames.
 * @param mask_sequence Vector of binary foreground masks.
 * @param bbox_sequence Vector of bounding boxes.
 * @param current_seq Metadata of the current sequence.
 * @param predicted_action Label assigned by the classifier.
 * @param dyn_feats Global kinematic features.
 * @param static_feats Spatial centroid features.
 * @param win_name Active OpenCV window title.
 * @param skip_visualization Flag set to true if user presses 'q'.
 * @param seq_index Index of the current sequence (modified by 'p'/'o').
 * @return bool True if sequence navigation was triggered.
 */
bool visualizeSequence(const std::vector<cv::Mat>& sequence_frames,
                       const std::vector<cv::Mat>& mask_sequence,
                       const std::vector<cv::Rect>& bbox_sequence,
                       const SequenceData& current_seq,
                       const std::string& predicted_action,
                       const SequenceFeatures& dyn_feats,
                       const StaticFeatures& static_feats,
                       const std::string& win_name,
                       bool& skip_visualization,
                       int& seq_index);

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
 * @brief Prints formatted dynamic features for sequence inspection.
 * @param seq_id Identifier string for the sequence (e.g., action_name/seq_name).
 * @param feats SequenceFeatures structure containing net displacement, average velocity, and stride peaks.
 * @return void
 */
void printSequenceFeaturesTable(const std::string& seq_id, const SequenceFeatures& feats);

/**
 * @brief Prints formatted static features for sequence inspection.
 * @param seq_id Identifier string for the sequence (e.g., action_name/seq_name).
 * @param feats StaticFeatures structure containing centroid metrics.
 * @return void
 */
void printStaticFeaturesTable(const std::string& seq_id, const StaticFeatures& feats);

/**
 * @brief Prints the final classification summary, confusion breakdown, and accuracy metrics.
 * @param total_processed Total number of video sequences evaluated.
 * @param total_correct Total number of sequences correctly classified.
 * @param class_total Map tracking the total sequence count for each action class.
 * @param class_correct Map tracking the correct prediction count for each action class.
 * @param error_log Vector of ErrorRecord structures detailing misclassified sequences.
 * @return void
 */
void printFinalReport(int total_processed, int total_correct,
                      const std::map<std::string, int>& class_total,
                      const std::map<std::string, int>& class_correct,
                      const std::vector<ErrorRecord>& error_log);



#endif // SEQUENCE_UTILS_H