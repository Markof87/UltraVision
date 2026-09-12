/**
 * @file ManipulatingService.cpp
 * @brief Implementation of the ManipulatingService class.
 * @author Marco Fabiani
 */

#include "ManipulatingService.h"
#include "SegmentationAlg.h"
#include "Bbox.h"

void ManipulatingService::manipulateFrame(const cv::Mat& inputFrame, cv::Mat& outputMask)
{
    if (inputFrame.empty()) return;
    SegmentationAlg::Segmentation(inputFrame, outputMask);

    cv::Mat bboxFrame = inputFrame.clone();
    Bbox::bbox_func(outputMask, bboxFrame);

    outputMask = bboxFrame;
    // outputMask = inputFrame.clone(); // For now, just clone the input frame to the output mask
}