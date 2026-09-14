/**
 * @file manipulating_service.cpp
 * @brief Implementation of the ManipulatingService class.
 * @author Marco Fabiani
 */

#include "manipulating_service.h"
#include "segmentation_alg.h"
#include "bbox.h"

void ManipulatingService::manipulateFrame(const cv::Mat& inputFrame, cv::Mat& outputMask)
{
    if (inputFrame.empty()) return;

    static SegmentationAlg seg_alg; 
    seg_alg.Segmentation(inputFrame, outputMask);

    cv::Mat bboxFrame = inputFrame.clone();
    Bbox::bbox_func(outputMask, bboxFrame);

    outputMask = bboxFrame;
    // outputMask = inputFrame.clone(); // For now, just clone the input frame to the output mask
}