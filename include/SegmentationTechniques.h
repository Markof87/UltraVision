/**
 * @file SegmentationTechniques.h
 * @brief Header file for the SegmentationTechniques class, including several techniques for segmentation.
 * @author Ivan Arduini
 * 
 */

#ifndef SEGMENTATION_TECHNIQUES_H
#define SEGMENTATION_TECHNIQUES_H

#include <opencv2/opencv.hpp>

/**
 * @class SegmentationTechniques
 * @brief Concrete implementation of the SegmentationTechniques class.
 * 
 * This static class provides functionality to implement several algorithms for segmentation.
 */

 class SegmentationTechniques
 {
    public:
        /**
         * @brief Static function to perform Otsu's thresholding on the input frame.
         * @param frame The input frame from which to work on.
         * @return A binary image representing the segmented output after applying Otsu's thresholding.
         */
        static cv::Mat otsu(const cv::Mat& frame);

        /**
         * @brief Static function to perform watershed segmentation on the input frame.
         * @param frame The input frame from which to work on.
         * @return A binary image representing the segmented output after applying watershed segmentation.
         */
        static cv::Mat watershed(const cv::Mat& frame);

        /**
         * @brief Static function to perform Mean Shift segmentation on the input frame.
         * @param frame The input frame from which to work on.
         * @return A binary image representing the segmented output after applying Mean Shift segmentation.
         */
        static cv::Mat meanshift(const cv::Mat& frame);
        
    private:
        /**
         * @brief Private constructor to prevent instantiation of the SegmentationTechniques class.
         */
        SegmentationTechniques() = default;
 };

 #endif // SEGMENTATION_TECHNIQUES_H