#ifndef BBOX_UTILS_H
#define BBOX_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>


/**
 * @brief Computes the centroid of a bounding box.
 * @param bbox The bounding box (cv::Rect).
 * @return cv::Point2f Centroid coordinates (x, y).
 */
cv::Point2f bbox_centroid(const cv::Rect& bbox);


/**
 * @brief Computes the net displacement (Euclidean distance) between the first and last valid centroid of a sequence.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return float Net displacement in pixels.
 */
float computeNetDisplacement(const std::vector<cv::Rect>& bbox_sequence);

/**
 * @brief Computes the average centroid velocity for a sequence of bounding boxes.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return float Average velocity in pixels/frame.
 */
float computeAverageSequenceVelocity(const std::vector<cv::Rect>& bbox_sequence);


/**
 * @brief Computes stride frequency by analyzing the width variations of the bounding box.
 *        Applies a 3-frame moving average filter to smooth out mask noise before peak detection.
 * @param bbox_sequence Vector of bounding boxes for the sequence.
 * @return int Number of detected peaks (stride cycles).
 */
int computeStrideFrequency_Width(const std::vector<cv::Rect>& bbox_sequence);


#endif // BBOX_UTILS_H