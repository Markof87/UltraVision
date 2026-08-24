#include "Segmentation.hpp"
#include <iostream>

// Function that applies bilateral filter first, followed by Otsu's thresholding
std::vector<cv::Mat> apply_bilateral_and_otsu(const std::vector<cv::Mat>& input_frames) {
    std::vector<cv::Mat> segmented_frames;

    for (const auto& frame : input_frames) {
        cv::Mat filtered_frame;
        cv::Mat otsu_frame;
        
        // 1. Apply bilateral filter to smooth noise while keeping edges sharp
        cv::bilateralFilter(frame, filtered_frame, 9, 75, 75);

        // 2. Apply Otsu's thresholding on the filtered frame (automatic global threshold)
        cv::threshold(filtered_frame, otsu_frame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        segmented_frames.push_back(otsu_frame);
    }

    return segmented_frames;
}

// Function that applies bilateral filter, Sobel, and Watershed segmentation
std::vector<cv::Mat> apply_bilateral_sobel_watershed(const std::vector<cv::Mat>& input_frames) {
    std::vector<cv::Mat> segmented_frames;

    for (const auto& frame : input_frames) {
        // 1. Apply bilateral filter
        cv::Mat filtered;
        cv::bilateralFilter(frame, filtered, 9, 75, 75);

        // 2. Apply Sobel to find gradients / edges
        cv::Mat grad_x, grad_y, grad;
        cv::Sobel(filtered, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(filtered, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad);
        cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 3. Create simple markers/threshold for Watershed based on Otsu or thresholding
        cv::Mat thresh;
        cv::threshold(filtered, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat markers;
        cv::connectedComponents(thresh, markers);
        
        // Add 1 so background is not 0, then markers are ready
        markers = markers + 1;

        // Watershed requires a 3-channel BGR image
        cv::Mat img3ch;
        cv::cvtColor(frame, img3ch, cv::COLOR_GRAY2BGR);

        // 4. Apply Watershed
        cv::watershed(img3ch, markers);

        // 5. Build final result mask where boundaries are marked (-1)
        cv::Mat result = cv::Mat::zeros(frame.size(), CV_8U);
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                if (markers.at<int>(i, j) == -1) {
                    result.at<uchar>(i, j) = 0;   // Boundary
                } else if (markers.at<int>(i, j) > 1) {
                    result.at<uchar>(i, j) = 255; // Foreground
                } else {
                    result.at<uchar>(i, j) = 0;   // Background
                }
            }
        }

        segmented_frames.push_back(result);
    }

    return segmented_frames;
}