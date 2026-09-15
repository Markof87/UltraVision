#ifndef BBOX_UTILS_H
#define BBOX_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "sequence_utils.h"

/**
 * @brief Classifies a dynamic sequence into running, jogging, or walking 
 *        using generalized kinematic thresholds optimized on dataset distributions.
 * @param feats SequenceFeatures structure containing global sequence metrics.
 * @return std::string Predicted dynamic action name.
 */
std::string classifyDynamicAction(const SequenceFeatures& feats);

/**
 * @brief Classifies a static sequence into boxing, handwaving, or handclapping
 *        using generalized spatial and centroid metrics optimized on dataset distributions.
 * @param feats StaticFeatures structure containing vertical reach and lateral asymmetry metrics.
 * @return std::string Predicted static action name.
 */
std::string classifyStaticAction(const StaticFeatures& feats);

/**
 * @brief Primary action classification router for the KTH dataset.
 *        Dispatches the sequence to dynamic or static classification routines
 *        based on global displacement and velocity bounds.
 * @param dyn_feats SequenceFeatures structure containing global kinematic metrics.
 * @param mask_sequence Vector of binary foreground/motion masks across all frames.
 * @param bbox_sequence Vector of tracked bounding boxes across all frames.
 * @return std::string Predicted action label for the entire video sequence.
 */
std::string classifySequence(const SequenceFeatures& dyn_feats,
                             const std::vector<cv::Mat>& mask_sequence,
                             const std::vector<cv::Rect>& bbox_sequence);


#endif // CLASSIFICATION_H