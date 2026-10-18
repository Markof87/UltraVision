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


    /**
     * @brief Processes an entire sequence of frames, computing masks, drawing bounding boxes,
     *        classifying the action, and returning annotated clones with the action label.
     * @param sequenceFrames Vector of raw frames belonging to a single sequence.
     * @param annotatedFrames Output vector storing cloned frames with bounding boxes and action text.
     * @return std::string The predicted action label.
     */
    std::string manipulateSequence(const std::vector<cv::Mat>& sequenceFrames, 
                               std::vector<cv::Mat>& annotatedFrames);
};

#endif // MANIPULATINGSERVICE_H