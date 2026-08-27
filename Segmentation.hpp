#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "DatasetManager.hpp" // Needed for SequenceData structure

// Function that applies bilateral filter first, followed by Otsu's thresholding on a single sequence
std::vector<cv::Mat> apply_bilateral_and_otsu(const std::vector<cv::Mat>& input_frames);


// Function that applies bilateral filter, Sobel gradients, and Watershed segmentation
std::vector<cv::Mat> apply_bilateral_sobel_watershed(const std::vector<cv::Mat>& input_frames);

// Function that applies Otsu's thresholding on a 2x2 grid and a morphological opening
std::vector<cv::Mat> apply_grid_otsu_opening(const std::vector<cv::Mat>& input_frames, int morph_kernel_size);

#endif // SEGMENTATION_HPP