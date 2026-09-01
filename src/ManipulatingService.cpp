/**
 * @file ManipulatingService.cpp
 * @brief Implementation of the ManipulatingService class.
 * @author Marco Fabiani
 */

#include "ManipulatingService.h"

void ManipulatingService::manipulateFrame(const cv::Mat& inputFrame, cv::Mat& outputMask)
{
    if (inputFrame.empty()) return;
    outputMask = inputFrame.clone(); // For now, just clone the input frame to the output mask
}