/**
 * @file FilterTechniques.cpp
 * @brief Implementation of the FilterTechniques class.
 * @author Matteo Longo
 */

#include "FilterTechniques.h"

cv::Mat FilterTechniques::sobel_filter(const cv::Mat& frame) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat filteredImage;

    // Calculate the x and y gradients using Sobel operator
    cv::Sobel(frame, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(frame, grad_y, CV_16S, 0, 1, 3);

    // Convert gradients to absolute values
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combine the two gradients
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, filteredImage);

    return filteredImage;
}

cv::Mat FilterTechniques::bilateral_filter(const cv::Mat& frame) {
    cv::Mat filteredImage;
    cv::bilateralFilter(frame, filteredImage, 9, 75, 75);
    return filteredImage;
}
