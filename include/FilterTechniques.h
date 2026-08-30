/**
 * @file FilterTechniques.h
 * @brief Header file for the FilterTechniques class, including several techniques for filtering.
 * @author Matteo Longo
 * 
 */

#ifndef FILTER_TECHNIQUES_H
#define FILTER_TECHNIQUES_H

#include <opencv2/opencv.hpp>

/**
 * @class FilterTechniques
 * @brief Concrete implementation of the FilterTechniques class.
 * 
 * This static class provides functionality to implement several algorithms for filtering.
 */

 class FilterTechniques
 {
    public:
        /**
         * @brief Static function to perform Sobel filtering on the input frame.
         * @param frame The input frame from which to work on.
         * @return A filtered image representing the output after applying Sobel filtering.
         */
        static cv::Mat sobel_filter(const cv::Mat& frame);
        
        /**
         * @brief Static function to perform bilateral filtering on the input frame.
         * @param frame The input frame from which to work on.
         * @return A filtered image representing the output after applying bilateral filtering.
         */
        static cv::Mat bilateral_filter(const cv::Mat& frame);
        
    private:
        /**
         * @brief Private constructor to prevent instantiation of the FilterTechniques class.
         */
        FilterTechniques() = default;
 };

 #endif // FILTER_TECHNIQUES_H