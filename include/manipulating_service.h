/**
 * @file ManipulatingService.h
 * @brief Header file for the ManipulatingService class, which provides functionality to manipulate a sequence of images.
 * @author Marco Fabiani
 */

#ifndef MANIPULATINGSERVICE_H
#define MANIPULATINGSERVICE_H

#include <opencv2/opencv.hpp>

/**
 * @class ManipulatingService
 * @brief A class that provides functionality to manipulate a sequence of images.
 */
class ManipulatingService
{
public:
    /**
     * @brief Manipulates a sequence of images.
     * @param inputFrame The input frame to be manipulated.
     * @param outputMask The output mask after manipulation.
     */
    static void manipulateFrame(const cv::Mat& inputFrame, cv::Mat& outputMask);
};

#endif // MANIPULATINGSERVICE_H