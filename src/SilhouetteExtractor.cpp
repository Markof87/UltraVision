/**
 * @file SilhouetteExtractor.cpp
 * @brief Implementation of the SilhouetteExtractor class.
 * @author Marco Fabiani
 */

#include "SilhouetteExtractor.h"

SilhouetteExtractor::SilhouetteExtractor() {
    // Constructor implementation (if needed)
}

SilhouetteExtractor::~SilhouetteExtractor() {
    // Destructor implementation (if needed)
}

SilhouetteExtractor::extractSilhouette(const cv::Mat& frame) {
    // You can choose the method you want to use for silhouette extraction here.
    // For example, you can call backgroundSubtraction or thresholding based on your needs.
    return backgroundSubtraction(frame);
}

SilhouetteExtractor::backgroundSubtraction(const cv::Mat& frame) {
    // Implement your background subtraction logic here.
    // This is a placeholder implementation.
    cv::Mat silhouette;
    // ... (background subtraction code)
    return silhouette;
}

SilhouetteExtractor::thresholding(const cv::Mat& frame) {
    // Implement your thresholding logic here.
    // This is a placeholder implementation.
    cv::Mat silhouette;
    // ... (thresholding code)
    return silhouette;
}