#include "SegmentationTechniques.h"

/**
 * @file SegmentationTechniques.cpp
 * @brief Implementation of the SegmentationTechniques class.
 * @author Ivan Arduini
 */

cv::Mat SegmentationTechniques::otsu(const cv::Mat& frame) {
    cv::Mat binaryImage;
    cv::threshold(frame, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binaryImage;
}

cv::Mat SegmentationTechniques::watershed(const cv::Mat& frame) {
    cv::Mat binaryImage;
    cv::Mat markers;
    cv::Mat grayFrame;

    // Convert the input frame to grayscale
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

    // Apply Otsu's thresholding to obtain a binary image
    cv::threshold(grayFrame, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Perform distance transform
    cv::Mat distTransform;
    cv::distanceTransform(binaryImage, distTransform, cv::DIST_L2, 5);
    cv::normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);

    // Threshold the distance transform to obtain markers
    cv::threshold(distTransform, markers, 0.5, 1.0, cv::THRESH_BINARY);
    markers.convertTo(markers, CV_32S);

    // Apply watershed algorithm
    cv::watershed(frame, markers);

    // Create a binary image for the segmented output
    binaryImage = markers > 1;

    return binaryImage;
}

cv::Mat SegmentationTechniques::meanshift(const cv::Mat& frame) {
    cv::Mat segmentedImage;
    cv::pyrMeanShiftFiltering(frame, segmentedImage, 21, 51);
    return segmentedImage;
}